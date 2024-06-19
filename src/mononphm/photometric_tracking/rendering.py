import torch
import torch.nn as nn
import pyvista as pv
import numpy as np
import trimesh
from time import time
from pytorch3d.transforms import so3_exp_map

from mononphm.photometric_tracking.utils import get_sphere_intersection


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-8  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device='cuda')
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device='cuda')
        u = torch.sort(u, dim=-1)[0]

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class RayMarcher(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_secant_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions,
                compute_non_convergent=False,
                gt_dist = None,
                debug : bool = False,
                samples = None,
                pose_params = None,
                ):

        batch_size, num_pixels, _ = ray_directions.shape
        t0 = time()

        if samples is None:
            sphere_intersections, mask_intersect = get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)
            #print(sphere_intersections.shape)
            #print(mask_intersect.shape)
            #print(mask_intersect.sum())
            #if gt_dist is not None:
            #    valid = ~torch.isinf(gt_dist)

            #    sphere_intersections[valid] = torch.stack([gt_dist[valid] - 0.2, gt_dist[valid] + 0.2], dim=-1)
            min_mask_points, min_mask_dist, sampled_points, net_values, steps = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask_intersect,
                                                                     sphere_intersections[..., 0], sphere_intersections[..., 1], pose_params=pose_params)
        else:
            n_rays = samples.shape[0]
            n_samples = samples.shape[1]
            sampled_points = samples
            net_values = []
            for pnts in torch.split(samples.reshape(-1, 3), 25000, dim=0):
                if pose_params is not None:
                    pnts = pose_params[2] * pnts @ so3_exp_map(pose_params[0]).squeeze().T + pose_params[1]
                net_values.append(sdf(pnts))
            net_values = torch.cat(net_values, dim=0).reshape(n_rays,n_samples, -1)
            steps = None




        return sampled_points, \
               net_values, steps


    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis, samples=None, pose_params=None,):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        #if mask.sum() == 351:
        #    print('hi')
        n_mask_points = mask.sum()

        n = self.n_steps
        steps = torch.linspace(0.0, 1.0, n).cuda()
        #steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis
        sample_dist = (mask_max_dis - mask_min_dis) / n
        t_rand = (torch.rand(steps.shape, device=steps.device) - 0.5)
        steps = steps+ t_rand * sample_dist
        #mids = .5 * (steps[:, 1:] + steps[:, :-1])
        #upper = torch.concat([mids, steps[:, -1:]], dim=-1)
        #lower = torch.concat([steps[:, :1], mids], dim=-1)
        #t_rand = torch.rand(steps.shape, device=steps.device)
        #steps = lower + (upper - lower) * t_rand

        reshaped = False
        if len(mask.shape) > 1:
            mask = mask.squeeze()
            reshaped = True
        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        if reshaped:
            mask = mask.unsqueeze(0)
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)



        mask_net_out_all = []
        for pnts in torch.split(points, 25000, dim=0):
            if pose_params is not None:
                pnts = pose_params[2] * pnts @ so3_exp_map(pose_params[0]).squeeze().T + pose_params[1]
            if len(pnts.shape) == 4:
                pnts = pnts.squeeze(0)
            mask_net_out_all.append(sdf(pnts))

        mask_net_out_all = torch.cat(mask_net_out_all).reshape(-1, n, mask_net_out_all[0].shape[-1])
        min_vals, min_idx = mask_net_out_all[..., 0].min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist, points, mask_net_out_all, steps.reshape(-1, n)


class VolumetricRenderer(nn.Module):
    def __init__(self, implicit_network_forward, conf):
        super().__init__()
        self.implicit_network = implicit_network_forward
        self.ray_tracer = RayMarcher(**conf['ray_tracer'])
        self.object_bounding_sphere = conf['ray_tracer']['object_bounding_sphere']

        self.register_parameter('neus_variance', torch.nn.Parameter(torch.tensor([0.6], requires_grad=True)))
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)




    def compute_SH_shading(self, normal_images, sh_coeff):
        '''
            normals: [nrays, nsamples, 3]
            sh_coeff: [1, 9, 3]
            self.constant_facto: [9]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, :, 0] * 0. + 1., N[:, :, 0], N[:, :, 1],
            N[:, :, 2], N[:, :, 0] * N[:, :, 1], N[:, :, 0] * N[:, :, 2],
            N[:, :, 1] * N[:, :, 2], N[:, :, 0] ** 2 - N[:, :, 1] ** 2, 3 * (N[:, :, 2] ** 2) - 1
        ],
            2)  # [nrays, nsamples, 9]
        sh = sh * self.constant_factor[None, None, :]
        shading = torch.sum(sh_coeff[None, :, :, :] * sh[:, :, :, None], 2)  # [bz, 9, 3,] before sum
        return shading # [bz, 3]

    def forward(self, input, condition, compute_non_convergent=False, skip_render=False, neus_variance=None, debug_plot=False, pose_params=None, use_SH = True, num_samples=32, sh_coeffs=None,
                w2c=None,):
        ray_dirs = input['ray_dirs']
        cam_loc = input['cam_loc']
        object_mask = input["object_mask"].reshape(-1)

        batch_size, num_pixels, _ = ray_dirs.shape

        if not skip_render:
            #self.implicit_network.eval()
            with torch.no_grad():
                points, net_values, steps = self.ray_tracer(sdf=lambda x: self.implicit_network(x, condition, include_color=False),
                                                                     cam_loc=cam_loc,
                                                                     object_mask=object_mask,
                                                                     ray_directions=ray_dirs,
                                                                     compute_non_convergent=compute_non_convergent,
                                                                     gt_dist=input['dist'] if 'dist' in input else None,
                                                                    pose_params=pose_params,
                                                                    )
                points = points.reshape(net_values.shape[0], net_values.shape[1], 3)

                vari = 0.3
                variance = torch.tensor(vari).cuda()
                inv_s = torch.exp(variance * 10)

                prev_cdf = torch.sigmoid(net_values[:, :-1, 0] * inv_s)
                next_cdf = torch.sigmoid(net_values[:, 1:, 0] * inv_s)

                p = prev_cdf - next_cdf
                c = prev_cdf

                alpha = ((p + 1e-8) / (c + 1e-8)).clip(0.0, 1.0)

                weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1], device='cuda'), 1. - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
                irrelevant_mask = weights < 1e-2
                irrelevant_mask = torch.zeros_like(irrelevant_mask)
                #print(f'Percent of relevant samples: {(torch.numel(irrelevant_mask)-irrelevant_mask.sum())/torch.numel(irrelevant_mask)}')

                weights[irrelevant_mask] = 0
                net_values[:, :-1, :][irrelevant_mask, :] = 0
                points[:, :-1, :][irrelevant_mask, :] = 0

                dists = None

            new_samples, z_samples = self.up_sample(cam_loc, ray_dirs, steps, net_values[:, :, 0], num_samples, inv_s)#sphere_radius=pose_params[2])

            if use_SH:
                dists = z_samples[..., 1:] - z_samples[..., :-1]
                dists = torch.cat([dists, dists[..., -1:]], -1)
                mid_z_vals = z_samples + dists * 0.5

                new_samples = cam_loc[:, None, :] + ray_dirs[0, :, None, :] * mid_z_vals[..., :,
                                                                              None]  # n_rays, n_samples, 3


            points, net_values, _ = self.ray_tracer(
                sdf=lambda x: self.implicit_network(x, condition, include_color=True, return_grad=use_SH),
                cam_loc=cam_loc,
                object_mask=object_mask,
                ray_directions=ray_dirs,
                compute_non_convergent=compute_non_convergent,
                gt_dist=input['dist'] if 'dist' in input else None,
                samples=new_samples,
                pose_params=pose_params,
                )

            points = points.reshape(net_values.shape[0], net_values.shape[1], 3) # nrays x samples_per_ray x 3

            if use_SH:
                true_cos = (ray_dirs[0, :, None, :] * net_values[..., -3:]).sum(-1, keepdim=True)

                nphm_space_normals = net_values[..., -3:] / torch.norm(net_values[..., -3:], dim=-1, keepdim=True)
                world_space_normals = nphm_space_normals @ so3_exp_map(pose_params[0]).squeeze() # apply inverse inverse wordl2model rotation
                if w2c is not None:
                    world_space_normals = world_space_normals @ w2c[:3, :3].T
                shading = self.compute_SH_shading(world_space_normals, sh_coeffs)

                variance = torch.tensor(neus_variance).cuda()
                inv_s = torch.exp(variance * 10)

                estimated_next_sdf = net_values[..., 0] + true_cos[..., 0] * dists * 0.5
                estimated_prev_sdf = net_values[..., 0] - true_cos[..., 0] * dists * 0.5

                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf

                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                alpha = alpha.clip(0.0, 1.0)
                weights = alpha * torch.cumprod(
                    torch.cat([torch.ones([alpha.shape[0], 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

                irrelevant_mask = weights < 1e-2
                irrelevant_mask = torch.zeros_like(irrelevant_mask)

                #print(
                #    f'Percent of relevant samples: {(torch.numel(irrelevant_mask) - irrelevant_mask.sum()) / torch.numel(irrelevant_mask)}')

                weights_sum = weights.sum(dim=-1, keepdim=True)
                weighted_depth = (z_samples[:, :] * weights).sum(dim=-1)

                color = ((shading[:, :, :] * ((net_values[:, :, 1:4] + 1) / 2) - 0.5) * 2 * weights[:, :, None]).sum(
                    dim=1)

                sdf_output = None,
                network_object_mask = None
                grad_theta = None
                dists = None

                rgb_values = color

            else:

                variance = torch.tensor(neus_variance).cuda()
                #print(self.neus_variance)
                inv_s = torch.exp(variance * 10)

                prev_sdf, next_sdf = net_values[:, :-1, 0], net_values[:, 1:, 0]
                prev_z_vals, next_z_vals = z_samples[:, :-1], z_samples[:, 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                cos_val = (next_sdf - prev_sdf + 1e-8) / (next_z_vals - prev_z_vals + 1e-8)

                cos_val = cos_val.clip(-1e3, 0.0)

                dist = (next_z_vals - prev_z_vals)
                prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
                next_esti_sdf = mid_sdf + cos_val * dist * 0.5
                prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
                next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
                alpha = (prev_cdf - next_cdf + 1e-8) / (prev_cdf + 1e-8)
                alpha = alpha.clip(0.0, 1.0)
                weights = alpha * torch.cumprod(
                    torch.cat([torch.ones([alpha.shape[0], 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]


                irrelevant_mask = weights < 1e-2
                irrelevant_mask = torch.zeros_like(irrelevant_mask)
                #print(
                #    f'Percent of relevant samples: {(torch.numel(irrelevant_mask) - irrelevant_mask.sum()) / torch.numel(irrelevant_mask)}')

                #weights[irrelevant_mask] = 0
                #net_values[:, :-1, :][irrelevant_mask, :] = 0
                #points[:, :-1, :][irrelevant_mask, :] = 0
                weights_sum = weights.sum(dim=-1, keepdim=True)
                weighted_depth = (z_samples[:, :-1] * weights).sum(dim=-1)
                # weights = weights/(weights_sum+1e-8)


                color = (( ((net_values[:, :-1, 1:4]+1)/2)-0.5)*2 * weights[:, :, None]).sum(dim=1)

                #weighted_points = (points[:, :-1] * weights[:, :, None]).sum(dim=1)
                sdf_output = None,
                network_object_mask = None
                grad_theta = None
                dists = None

                rgb_values = color

        else:
            points = None
            rgb_values = None
            sdf_output=None
            network_object_mask = None
            object_mask = None
            dists = None
            grad_theta = None
            weights_sum = None

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'dists': dists,
            'weights_sum': weights_sum,
            'weighted_depth': weighted_depth,
        }

        return output


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, sphere_radius=1.0):
        """
        Up sampling give a fixed inv_s
        """

        batch_size, n_samples = z_vals.shape
        if rays_d.shape[1] != z_vals.shape[0]:
            print('hi')
        pts = rays_o[:, None, :] + rays_d[0, :, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < sphere_radius) | (radius[:, 1:] < sphere_radius)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf + 1e-8) / (next_z_vals - prev_z_vals + 1e-8)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device='cuda'), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-8) / (prev_cdf + 1e-8)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        new_samples = rays_o[:, None, :] + rays_d[0, :, None, :] * z_samples[..., :, None]  # n_rays, n_samples, 3


        return new_samples, z_samples

