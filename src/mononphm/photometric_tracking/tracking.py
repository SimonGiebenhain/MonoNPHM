import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import mediapy

from pytorch3d.transforms import so3_exp_map, so3_log_map

import matplotlib.pyplot as plt

from mononphm import env_paths
from mononphm.photometric_tracking.rendering import VolumetricRenderer
from mononphm.models.reconstruction import get_logits, get_vertex_color
from mononphm.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from mononphm.models.iterative_root_finding import search
from mononphm.models.diff_operators import jac
from mononphm.utils.render_utils import project_points_torch
from mononphm.photometric_tracking.data_loading import prepare_data



ANCHOR_iBUG68_pairs_65 = np.array([
    [0, 0], #left upmost jaw
    [1, 16], #right upmost jaw
    [38, 2], # jaw
    [39, 14], # jaw
    [2, 4], # jaw
    [3, 12], # jaw
    [4, 6], # jaw
    [5, 10], # jaw
    [60, 8], # chin

    [10, 31], # nose
    [11, 35], # nose
    [62, 30], # nose tip
    [61, 27], # nose top

    [6, 17], #l eyebrow outer
    [7, 26], # r eyebrow outer
    #[8, 19], # l eyebrow
    #[9, 24], # r eyebrow

    [12, 36], # l eye outer,
    [13, 45], # r eye outer,
    [14, 39], # l eye inner,
    [15, 42], # r eye inner,

    [16, 48], # l mouth corner
    [17, 54], # r mouth corner
    [18, 50], # l mouth top
    [19, 52], # r mouth top
    [20, 58], # l mouth bottom
    [21, 56], # r mouth bottom,
    [44, 49], # mouth
    [45, 53],# mouth
    [46, 59], # mouth
    [47, 55],# mouth
    [48, 38],#eye
    [49, 43],#eye
    [50, 41],#eye
    [51, 46],#eye
    [52, 21], # eye brow inner
    [53, 22] # eye brow inner

])



class ImageSpaceLosses(nn.Module):
    def __init__(self, rgb_weight,  mask_weight, lambdas_reg, reg_weight_expr,
                 ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.mask_weight = mask_weight
        self.lambdas_reg = lambdas_reg
        self.reg_weight_expr = reg_weight_expr
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask, valid = None):
        valid = valid.squeeze()
        if network_object_mask is None:
            mouth_interior = ~valid[object_mask]
            rgb_loss = torch.abs(rgb_values[object_mask] - rgb_gt[0, object_mask, :]) / float(object_mask.shape[0])
            rgb_loss[mouth_interior] = rgb_loss[mouth_interior] / 25
            rgb_loss = rgb_loss.sum()
        else:
            if (network_object_mask & object_mask).sum() == 0:
                return torch.tensor(0.0).cuda().float()
            mask = network_object_mask & object_mask
            mouth_interior = ~valid[mask]

            rgb_loss = torch.abs(rgb_values[mask]- rgb_gt[0, mask, :]) / float(object_mask.shape[0])
            rgb_loss[mouth_interior] = rgb_loss[mouth_interior] / 25
            rgb_loss = rgb_loss.sum()


        return rgb_loss


    def get_jaccard_distance_loss(self, y_pred, y_true, smooth=5):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        y_pred = y_pred.squeeze().clamp(0, 1)

        intersection = torch.sum(torch.abs(y_true * y_pred))
        sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth


    def forward(self, model_outputs, ground_truth, geo_loss3d=None, rgb_loss3d=None):

        if self.rgb_weight > 0:
            rgb_gt = ground_truth['rgb']
            network_object_mask = model_outputs['network_object_mask']
            facer_mask = model_outputs['object_mask']

            rgb_loss_mask = ~((facer_mask == 3) | (facer_mask == 0)) #| (facer_mask == facer_mask.max()))
            foreground_mask = rgb_loss_mask #| (facer_mask == 1)
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, rgb_loss_mask, valid=ground_truth['mm']==0)

            hair_region = facer_mask == 14
            n_hair = hair_region.sum()
            nn_hair = (~hair_region).sum() > 0
            if n_hair > 0:
                mask_loss_hair_region = self.get_jaccard_distance_loss(model_outputs['weights_sum'][hair_region], foreground_mask[hair_region])
            else:
                mask_loss_hair_region = 0
            if nn_hair:
                mask_loss = self.get_jaccard_distance_loss(model_outputs['weights_sum'], foreground_mask)
            else:
                mask_loss = 0
            mask_loss = (n_hair*mask_loss + nn_hair*mask_loss_hair_region/5)/(nn_hair+n_hair)

            reg_loss = 0
            for k in model_outputs['reg_loss'].keys():
                reg_loss += self.lambdas_reg[k] * model_outputs['reg_loss'][k]
            reg_loss_expr = model_outputs['reg_loss_expr']


            loss = self.rgb_weight * rgb_loss + \
                   self.mask_weight * mask_loss + \
                    reg_loss + \
                    self.reg_weight_expr * reg_loss_expr
        else:
            rgb_loss = torch.tensor([0])
            mask_loss = torch.tensor([0])
            reg_loss_expr = model_outputs['reg_loss_expr']
            reg_loss = 0
            for k in model_outputs['reg_loss'].keys():
                reg_loss += self.lambdas_reg[k] * model_outputs['reg_loss'][k]
            loss = reg_loss + self.reg_weight_expr * reg_loss_expr

        return_dict =  {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
            'reg_loss_expr': reg_loss_expr,
        }

        return_dict.update(model_outputs['reg_loss'])
        return return_dict



def render_image(idr, expression, condition, sh_coeffs, in_dict, n_batch_points=2000, w=None, h=None, scale_uniformly = False, variance=None, pose_params=None, use_sh=False):
    idr.eval()
    #idr.train()
    torch.cuda.empty_cache()
    if use_sh:
        n_batch_points = 500 # for SH cannot handle large batch sizes
    ray_dir_split = torch.split(in_dict['ray_dirs'], n_batch_points, dim=1)
    object_mask_split = torch.split(in_dict['object_mask'], n_batch_points, dim=1)
    color_list = []
    list_object_mask = []
    list_network_object_mask = []
    list_sdf_outputs = []
    list_weights_sum = []
    list_depths = []
    for chunk_i, (ray_dirs, object_masks) in enumerate(zip(ray_dir_split, object_mask_split)):
        if use_sh:
            #with torch.no_grad(): # needs to be removed fro SH
                cur_in_dict = {'ray_dirs': ray_dirs, 'object_mask': object_masks, 'cam_loc': in_dict['cam_loc']}
                out_dict = idr(cur_in_dict, condition, compute_non_convergent=True, neus_variance=variance,
                               pose_params=[pose_params[0], pose_params[1], pose_params[2]],
                               sh_coeffs=sh_coeffs)#, debug_plot=chunk_i==0)
                print('hi')

                torch.cuda.empty_cache()

                color = out_dict['rgb_values'].detach()
                color = color.squeeze()
                color_list.append(color.squeeze(0).detach().cpu())
                if out_dict['sdf_output'][0] is None:
                    list_sdf_outputs.append(torch.zeros([out_dict['rgb_values'].shape[0]]))
                    list_network_object_mask.append(torch.zeros([out_dict['rgb_values'].shape[0]], dtype=torch.bool))

                else:
                    list_sdf_outputs.append(out_dict['sdf_output'].squeeze(0).detach().cpu())
                    list_network_object_mask.append(out_dict['network_object_mask'].squeeze(0).detach().cpu())
                list_weights_sum.append(out_dict['weights_sum'].squeeze(0).detach().cpu())
                list_depths.append(out_dict['weighted_depth'].squeeze(0).detach().cpu())


                list_object_mask.append(out_dict['object_mask'].squeeze(0).detach().cpu())
                del out_dict, color
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():  # needs to be removed fro SH
                cur_in_dict = {'ray_dirs': ray_dirs, 'object_mask': object_masks, 'cam_loc': in_dict['cam_loc'],
                               }
                out_dict = idr(cur_in_dict, expression, compute_non_convergent=True, neus_variance=variance,
                               pose_params=[pose_params[0], pose_params[1], pose_params[2]])  # , debug_plot=chunk_i==0)
                print('hi')

                torch.cuda.empty_cache()

                color = out_dict['rgb_values'].detach()
                color = color.squeeze()
                color_list.append(color.squeeze(0).detach().cpu())
                if out_dict['sdf_output'][0] is None:
                    list_sdf_outputs.append(torch.zeros([out_dict['rgb_values'].shape[0]]))
                    list_network_object_mask.append(torch.zeros([out_dict['rgb_values'].shape[0]], dtype=torch.bool))

                else:
                    list_sdf_outputs.append(out_dict['sdf_output'].squeeze(0).detach().cpu())
                    list_network_object_mask.append(out_dict['network_object_mask'].squeeze(0).detach().cpu())
                list_weights_sum.append(out_dict['weights_sum'].squeeze(0).detach().cpu())
                list_depths.append(out_dict['weighted_depth'].squeeze(0).detach().cpu())


                list_object_mask.append(out_dict['object_mask'].squeeze(0).detach().cpu())
                del out_dict, color
                torch.cuda.empty_cache()
    color = np.concatenate(color_list, axis=0)
    sdf = np.concatenate(list_sdf_outputs, axis=0)
    object_mask = np.concatenate(list_object_mask, axis=0)
    network_object_mask = np.concatenate(list_network_object_mask, axis=0)
    _weights_sum = np.concatenate(list_weights_sum, axis=0)
    depths = np.concatenate(list_depths, axis=0)
    color = np.reshape(color, [h, w, 3])
    sdf = np.reshape(sdf, [h, w])
    object_mask = np.reshape(object_mask, [h, w]).astype(np.float32)
    network_object_mask = np.reshape(network_object_mask, [h, w]).astype(np.float32)
    weights_sum = np.reshape(_weights_sum, [h, w]).astype(np.float32)
    weights_img = ((np.tile(np.reshape(_weights_sum, [h, w]).astype(np.float32)[:, :, np.newaxis], [1, 1, 3])/1)*255).astype(np.uint8)

    print('DEPTH:', depths.min(), depths.max())

    #depths = ((np.clip((np.tile(np.reshape(depths, [h, w]).astype(np.float32)[:, :, np.newaxis], [1, 1, 3]) - 2.25)/ (2.92-2.25), 0, 1) )*255).astype(np.uint8)

    min_depth = depths.min()

    depths = (np.tile(np.reshape(depths, [h, w]).astype(np.float32)[:, :, np.newaxis], [1, 1, 3]) - min_depth)
    max_depth = depths.max()
    depths = depths / max_depth
    depths = (depths*255).astype(np.uint8)
#
    if not scale_uniformly:
        color *= np.array([62.06349782, 52.41366313, 48.37649288])
        color += np.array([109.23604821, 98.02477547, 87.84371274])
    else:
        color = (color + 1) / 2 * 255

    color = np.clip(color, 0, 255)
    color = color.astype(np.uint8)

    object_mask *= 255
    network_object_mask *= 255
    object_mask = object_mask.astype(np.uint8)
    network_object_mask = network_object_mask.astype(np.uint8)


    I_o_mask = Image.fromarray(object_mask)
    I_pred_o_mask = Image.fromarray(network_object_mask)
    I = Image.fromarray(color)
    idr.train()
    return I, I_o_mask, I_pred_o_mask, sdf, weights_sum, weights_img, depths





class SimpleDataManager:
    def __init__(self,
                 expressions,
                 seq_name,
                 cfg,
                 num_views : int = 1,
                 downsampling_factor : float = 1/6,
                 intrinsics_provided : bool = True,
                 ):
        self.cfg = cfg
        self.num_views = num_views
        self.expressions = expressions
        self.downsampling_factor = downsampling_factor
        self.intrinsics_provided = intrinsics_provided
        # generate ray-based input tensors
        rgbs, masks, view_dirss, cam_poss, ws, hs, gt_imgs, c2ws, intrinsics = [], [], [], [], [], [], [], [], []
        lms = []
        mouth_interiors = []

        for expression in expressions:

            res_cleaned = prepare_data(timestep=expression, seq_name=seq_name,
                                       downsample_factor=downsampling_factor,
                                       intrinsics_provided=intrinsics_provided)
            rgbs.append(res_cleaned['rgb'])
            masks.append(res_cleaned['segmentation_mask'])
            view_dirss.append(res_cleaned['view_dir'])
            cam_poss.append(res_cleaned['cam_pos'])
            ws.append(res_cleaned['width'])
            hs.append(res_cleaned['height'])
            c2ws.append(res_cleaned['w2c'])
            intrinsics.append(res_cleaned['intrinsics'])
            lms.append(res_cleaned['landmarks_2d'])
            mouth_interiors.append(res_cleaned['mouth_interior_mask'])


        all_detected_lms = np.load(f'{env_paths.DATA_TRACKING}/{seq_name}/pipnet/test.npy')

        self.change = np.nanmean(np.square((all_detected_lms[1:, ...] - all_detected_lms[:-1, ...])), axis=(1, 2))
        self.mean_change = np.mean(self.change)

        self.N_rays = [[rgb.shape[0] for rgb in expression_rgbs] for expression_rgbs in rgbs]

        gt_rgb = [[rgb[:, :].unsqueeze(0).float() for rgb in expression_rgbs] for expression_rgbs in rgbs]
        gt_mouth_interiors = [[mm.unsqueeze(0).float() for mm in cur_mm] for cur_mm in mouth_interiors]
        gt_mask = [[mask[:].unsqueeze(0).unsqueeze(-1) for mask in expression_masks] for expression_masks in masks]
        gt_view_dir = [[view_dirs[:, :].unsqueeze(0).float() for view_dirs in expression_view_dirss] for
                       expression_view_dirss in view_dirss]
        gt_cam_pos = [[cam_pos.unsqueeze(0).float() for cam_pos in expression_cam_poss] for expression_cam_poss in
                      cam_poss]


        self.full_in_dict = {
            'ray_dirs': gt_view_dir,
            'cam_loc': gt_cam_pos,
            'object_mask': gt_mask,
            'rgb': gt_rgb,
            'mouth_interior': gt_mouth_interiors,
        }

        self.hs = hs
        self.ws = ws

        self.rgbs = [[I.reshape(hs[j][i], ws[j][i], 3) for i, I in enumerate(Is)] for j, Is in enumerate(rgbs)]
        self.mms = [[I.reshape(hs[j][i], ws[j][i]) for i, I in enumerate(Is)] for j, Is in enumerate(mouth_interiors)]
        self.view_dirss = [[I.reshape(hs[j][i], ws[j][i], 3) for i, I in enumerate(Is)] for j, Is in enumerate(view_dirss)]
        self.masks = [[I.reshape(hs[j][i], ws[j][i]) for i, I in enumerate(Is)] for j, Is in enumerate(masks)]

        self.cam_poss = cam_poss
        self.lms = lms

        self.intrinsics = intrinsics
        self.c2ws = c2ws


    def get_random_input(self, ):
        current_expression = np.random.randint(0, len(self.expressions))
        rnd_idx = np.random.randint(0, self.num_views)

        # subsample rays, add batch_dim, push to GPU
        selected_rays_w = torch.randint(0 + 2, self.ws[current_expression][rnd_idx] - 2,
                                        [10000])
        selected_rays_h = torch.randint(0 + 2, self.hs[current_expression][rnd_idx] - 2,
                                        [10000])
        gt_rgb = self.rgbs[current_expression][rnd_idx][selected_rays_h, selected_rays_w, :].unsqueeze(0).cuda().float()
        gt_mm = self.mms[current_expression][rnd_idx][selected_rays_h, selected_rays_w].unsqueeze(0).cuda().float()

        gt_mask = self.masks[current_expression][rnd_idx][selected_rays_h, selected_rays_w].unsqueeze(0).unsqueeze(-1).cuda()
        gt_view_dir = self.view_dirss[current_expression][rnd_idx][selected_rays_h, selected_rays_w, :].unsqueeze(
            0).cuda().float()
        gt_cam_pos = self.cam_poss[current_expression][rnd_idx].unsqueeze(0).cuda().float()

        foreground_index = torch.nonzero(gt_mask)[:, 1]
        selected_forground = torch.randint(0, foreground_index.shape[0], [int(self.cfg['opt']['rays_per_batch'] * 0.75)])
        selected_others = torch.randint(0, 10000,
                                        [self.cfg['opt']['rays_per_batch'] - int(self.cfg['opt']['rays_per_batch'] * 0.75)])

        gt_rgb = torch.cat([gt_rgb[:, selected_forground, :], gt_rgb[:, selected_others, :]], dim=1)
        gt_mm = torch.cat([gt_mm[:, selected_forground], gt_mm[:, selected_others]], dim=1)
        gt_mask = torch.cat([gt_mask[:, selected_forground, :], gt_mask[:, selected_others, :]], dim=1)
        gt_view_dir = torch.cat([gt_view_dir[:, selected_forground, :], gt_view_dir[:, selected_others, :]], dim=1)


        in_dict = {
            'ray_dirs': gt_view_dir,
            'cam_loc': gt_cam_pos,
            'object_mask': gt_mask,
            'rgb': gt_rgb,
            'mm': gt_mm,
        }
        return current_expression, rnd_idx, in_dict


class Tracker:
    def __init__(self,
                 subject,
                 expressions,
                 cfg,
                 net,
                 out_dir,
                 lr_scale,
                 fix_id,
                 num_views : int = 1,
                 fine_tune_id : bool = False,
                 out_dir_stage1 = None,
                 ):
        self.lr_scale = lr_scale
        self.lr_scale_pose = lr_scale
        if self.lr_scale < 1:
            self.lr_scale_pose /= 10  # 0
        self.subject = subject
        self.expressions = expressions
        self.out_dir = out_dir
        self.out_dir_stage1 = out_dir_stage1
        self.num_views = num_views
        self.fine_tune_id = fine_tune_id
        self.net = net.monoNPHM
        self.fix_id = fix_id
        self.rec_conf = cfg['reconstruction']

        self.anchors_posed = {}

        if len(expressions) > 1:
            expression = 'ALL'
        else:
            expression = expressions[0]

        self.expression = expression

        if fine_tune_id:
            self.exp_dir = f'{out_dir}/szage2/{subject}/'
        else:
            self.exp_dir = f'{out_dir}/stage1/{subject}/{expression:05d}/'

        if not fine_tune_id:
            while os.path.exists(f'{self.exp_dir}/z_geo.npy'):

                expression += 1
                self.expression = expression
                self.expressions = [expression]
                expressions = [expression]

                self.exp_dir = f'{out_dir}/stage1/{subject}/{expression:05d}//'

        if not fine_tune_id:
            self.exp_dir_prev = f'{out_dir}/stage1/{subject}/{expression-1:05d}/'
            self.progress_dir = f'{out_dir}/stage1/{subject}/{expression:05d}/progress/'

        else:
            self.exp_dir_prev = None
            self.progress_dir = f'{out_dir}/stage2/{subject}/progress/'
        if fine_tune_id:
            self.exp_dir = f'{out_dir}/stage2/{subject}/'
            self.progress_dir = f'{out_dir}/stage2/{subject}/progress/'
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)

        self.idr = VolumetricRenderer(net, cfg).cuda()
        self.loss_function = ImageSpaceLosses(rgb_weight=cfg['opt']['lambda_rgb'],
                                              mask_weight=cfg['opt']['lambda_mask'],
                                              lambdas_reg=cfg['opt']['lambdas_reg'],
                                              reg_weight_expr=cfg['opt']['lambda_reg_expr'],
                                              )

        if not hasattr(net.monoNPHM.id_model, 'n_anchors'):
            net.monoNPHM.id_model.n_anchors = 65

        anchor_loss_scale = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, #jaw
             100, #chin
             1, 1, 1, 1, # nose
             1, 1, #eye brow outer
             1, 1, 1, 1, # eye outer/inner
             100, 100, 100, 100, 100, 100, # mouth
             100, 100, 100, 100, # mouth
             100, 100, 100, 100, # eye
             1, 1 # not sure
             ])

        self.anchor_loss_scale = torch.from_numpy(anchor_loss_scale).cuda()

        self.lm_inds = torch.from_numpy(np.array([2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
                                             3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
                                             3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
                                             3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
                                             1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
                                             1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
                                             3509, 2786])).cuda()


        self.progress_images =  {e: {i: [] for i in range(num_views)} for e in range(len(expressions))}


        if fine_tune_id:
            self.setup_latent_codes_stage2()
        else:
            self.setup_latent_codes_stage1()

        self.setup_optimization()


    def setup_latent_codes_stage1(self, ):

        # load latent codes from previous steps
        if os.path.exists(self.exp_dir_prev):
            lat_rep_shape = torch.from_numpy(np.load(self.exp_dir_prev + 'z_geo.npy')).cuda()
            lat_rep_app = torch.from_numpy(np.load(self.exp_dir_prev + 'z_app.npy')).cuda()
            lat_rep_exp = torch.from_numpy(np.load(self.exp_dir_prev + f'z_exp.npy')).cuda().squeeze()
            lat_rep_shape.requires_grad = False
            lat_rep_app.requires_grad = False
            lat_rep_exp.requires_grad = True # expression code only serves as initialization
        else:
            if hasattr(self.net.id_model, 'lat_dim_loc_geo'):
                lat_rep_shape = torch.zeros([1, 1, self.net.id_model.lat_dim_glob + self.net.id_model.lat_dim_loc_geo * (self.net.id_model.n_anchors + 1)]).cuda()
            else:
                lat_rep_shape = torch.zeros([1, 1, self.net.id_model.lat_dim]).cuda()

            lat_rep_shape.requires_grad = True

            if hasattr(self.net.id_model, 'lat_dim_loc_app'):
                lat_rep_app = torch.zeros(
                    [1, 1, self.net.id_model.lat_dim_glob + self.net.id_model.lat_dim_loc_app * (self.net.id_model.n_anchors + 1)]).cuda()
            else:
                lat_rep_app = torch.zeros([1, 1, self.net.id_model.lat_dim_app]).cuda()
            lat_rep_app.requires_grad = True

            lat_rep_exp = torch.zeros([1, 1, self.net.ex_model.lat_dim_expr]).cuda()
            lat_rep_exp.requires_grad = True
            lat_rep_exp = lat_rep_exp.squeeze()

        assert len(lat_rep_exp.shape) == 1
        expr_codebook = torch.nn.Embedding(num_embeddings=len(self.expressions), embedding_dim=lat_rep_exp.shape[0], sparse=True)
        expr_codebook.weight = torch.nn.Parameter(lat_rep_exp.unsqueeze(0).repeat(len(self.expressions), 1))

        latent_code = {'geo': lat_rep_shape,
                       'app': lat_rep_app}

        self.expr_codebook = expr_codebook
        self.latent_code = latent_code

        if os.path.exists(self.exp_dir_prev):
            if os.path.exists(self.exp_dir_prev + 'colorA.npy'):
                colorA = np.load(self.exp_dir_prev + 'colorA.npy')
                colorb = np.load(self.exp_dir_prev + 'colorb.npy')
            else:
                colorA, colorb = None, None
            if os.path.exists(self.exp_dir_prev + 'scale.npy'):
                scale = np.load(self.exp_dir_prev + 'scale.npy')
                rot = np.load(self.exp_dir_prev + 'rot.npy')
                trans = np.load(self.exp_dir_prev + 'trans.npy')
                params_pose = [rot, trans, scale]
                if len(self.expressions) > 1:
                    params_pose = [params_pose] + list([None, ] * (len(self.expressions) - 1))
                else:
                    params_pose = [params_pose]
            else:
                params_pose = None
            if os.path.exists(self.exp_dir_prev + 'sh_coeffs.npy'):
                sh_coeffs = np.load(self.exp_dir_prev + 'sh_coeffs.npy')
            else:
                sh_coeffs = None
            self.colorA = colorA
            self.colorb = colorb
            self.params_pose = params_pose
            self.sh_coeffs = torch.from_numpy(sh_coeffs).cuda()
        else:
            self.colorA = None
            self.colorb = None
            self.params_pose = []

            sh_coeffs = torch.zeros(1, 9, 3).float().cuda()
            sh_coeffs[:, 0, :] = np.sqrt(4 * np.pi)
            sh_coeffs = sh_coeffs.cuda()
            self.sh_coeffs = sh_coeffs
            self.sh_coeffs.requires_grad = True


    def setup_latent_codes_stage2(self,):
        exp_dir_start = f'{self.out_dir_stage1}/{self.subject}/{self.expressions[0]:05d}/'
        lat_rep_shape = torch.from_numpy(np.load(exp_dir_start + 'z_geo.npy')).cuda()
        lat_rep_app = torch.from_numpy(np.load(exp_dir_start + 'z_app.npy')).cuda()
        lat_rep_shape.requires_grad = True
        lat_rep_app.requires_grad = True
        scale = np.load(exp_dir_start + 'scale.npy')

        lat_reps_exp = []
        pose_params = []
        for e in self.expressions:
            exp_dir = f'{self.out_dir_stage1}/{self.subject}/{e:05d}/'
            lat_rep_exp = torch.from_numpy(np.load(exp_dir + f'z_exp.npy')).cuda()
            lat_rep_exp.requires_grad = True
            lat_reps_exp.append(lat_rep_exp.squeeze())


            rot = np.load(exp_dir + 'rot.npy')
            trans = np.load(exp_dir + 'trans.npy')
            params_pose = [rot, trans, scale]
            pose_params.append(params_pose)
        expr_codebook = torch.nn.Embedding(num_embeddings=len(lat_reps_exp), embedding_dim=lat_reps_exp[0].shape[0],
                                           sparse=True)
        lat_reps_exp = torch.stack(lat_reps_exp, dim=0)
        expr_codebook.weight = torch.nn.Parameter(lat_reps_exp)
        self.expr_codebook = expr_codebook

        latent_code = {'geo': lat_rep_shape,
                       'app': lat_rep_app}

        colorA = np.load(exp_dir_start + 'colorA.npy')
        colorb = np.load(exp_dir_start + 'colorb.npy')
        if os.path.exists(exp_dir_start + 'sh_coeffs.npy'):
            sh_coeffs = np.load(exp_dir_start + 'sh_coeffs.npy')
        else:
            sh_coeffs = None

        self.latent_code = latent_code
        self.colorb = colorb
        self.colorA = colorA
        self.sh_coeffs = torch.from_numpy(sh_coeffs).cuda()
        self.sh_coeffs.required_grad = True

        self.params_pose = pose_params

        #TODO process pose params properly ?? or not required?!


    def setup_optimization(self):
        if self.fix_id and not self.fine_tune_id:
            params = []
        else:
            self.latent_code['geo'].requires_grad = True
            self.latent_code['app'].requires_grad = True
            params = [self.latent_code['geo'], self.latent_code['app']]
        params_expr = self.expr_codebook.parameters()
        if self.colorA is not None:
            if not isinstance(self.colorA, torch.Tensor):
                self.colorA = torch.from_numpy(self.colorA).cuda()
                self.colorb = torch.from_numpy(self.colorb).cuda()
        else:
            colorA = torch.eye(3, device='cuda')
            colorb = torch.zeros([1, 3], device='cuda')
            colorA.requires_grad = True
            colorb.requires_grad = True
            params.append(colorA)
            params.append(colorb)
            self.colorA = colorA
            self.colorb = colorb
        if len(self.params_pose) > 0:
            if len(self.params_pose) == 3 and len(self.expressions) != 3:
                self.params_pose = [self.params_pose]
            params_pose_individual = []
            for i_ex in range(len(self.params_pose)):
                if self.params_pose[i_ex] is None:
                    rot_params = so3_log_map(torch.eye(3).unsqueeze(0)).repeat(1, 1).float().cuda()
                    trans_params = torch.zeros([1, 3]).float().cuda()
                    scale_param = torch.ones([1]).float().cuda()
                    _params_pose = [rot_params, trans_params, scale_param]
                else:
                    if len(self.params_pose[i_ex][0].shape) == 3:
                        _params_pose = [
                            so3_log_map(torch.from_numpy(self.params_pose[i_ex][0])).float().cuda(),
                            torch.from_numpy(self.params_pose[i_ex][1]).squeeze().unsqueeze(0).float().cuda(),
                            torch.from_numpy(self.params_pose[i_ex][2]).float().cuda(),
                        ]
                    else:
                        _params_pose = [
                            so3_log_map(torch.from_numpy(self.params_pose[i_ex][0]).unsqueeze(0)).float().cuda(),
                            torch.from_numpy(self.params_pose[i_ex][1]).squeeze().unsqueeze(0).float().cuda(),
                            torch.from_numpy(self.params_pose[i_ex][2]).float().cuda(),
                        ]
                params_pose_individual.append(_params_pose)

            params_pose_rot = torch.cat([params_pose_individual[i][0] for i in range(len(self.expressions))], dim=0)
            params_pose_trans = torch.cat([params_pose_individual[i][1] for i in range(len(self.expressions))], dim=0)
            self.scale_param = params_pose_individual[0][2]
            self.scale_param.requires_grad = False
            self.params_pose = [params_pose_rot, params_pose_trans]

        else:
            rot_params = so3_log_map(torch.eye(3).unsqueeze(0)).repeat(len(self.expressions), 1).float().cuda()
            trans_params = torch.zeros([len(self.expressions), 3]).float().cuda()
            self.scale_param = torch.ones([1]).float().cuda()
            rot_params.requires_grad = True
            trans_params.requires_grad = True
            self.scale_param.requires_grad = False
            self.params_pose.append(rot_params)
            self.params_pose.append(trans_params)
        params.append(self.idr.neus_variance)


        if self.fine_tune_id:
            self.opt = torch.optim.Adam(params=params, lr=0.0002)
            self.opt_expr = torch.optim.SparseAdam(params=params_expr, lr=0.0001)

            if self.fix_id and not self.fine_tune_id:
                self.sh_coeffs.requires_grad = False
            else:
                self.sh_coeffs.requires_grad = True

            self.opt_sh = torch.optim.Adam(params=[self.sh_coeffs], lr=0.0001)

            self.params_pose = torch.cat(self.params_pose, dim=-1)
            pose_codebook_rot = torch.nn.Embedding(num_embeddings=self.params_pose.shape[0],
                                               embedding_dim=3, sparse=True)
            pose_codebook_trans = torch.nn.Embedding(num_embeddings=self.params_pose.shape[0],
                                                   embedding_dim=3, sparse=True)
            pose_codebook_rot.weight = torch.nn.Parameter(self.params_pose[:, :3].contiguous())
            pose_codebook_trans.weight = torch.nn.Parameter(self.params_pose[:, 3:6].contiguous())
            self.optim_pose_rot = torch.optim.SparseAdam(params=pose_codebook_rot.parameters(), lr=0.0003)
            self.optim_pose_trans = torch.optim.SparseAdam(params=pose_codebook_trans.parameters(), lr=0.0001)
            if self.fix_id and not self.fine_tune_id:
                self.scale_param.requires_grad = False
            self.optim_scale = torch.optim.Adam(params=[self.scale_param], lr=0.00003)
        else:
            self.opt = torch.optim.Adam(params=params, lr=0.0005 * self.lr_scale)
            self.opt_expr = torch.optim.SparseAdam(params=params_expr,
                                              lr=0.0002 * self.lr_scale)

            if self.fix_id and not self.fine_tune_id:
                self.sh_coeffs.requires_grad = False
            else:
                self.sh_coeffs.requires_grad = True
            self.opt_sh = torch.optim.Adam(params=[self.sh_coeffs], lr=0.05 * self.lr_scale)

            self.params_pose = torch.cat(self.params_pose, dim=-1)
            pose_codebook_rot = torch.nn.Embedding(num_embeddings=self.params_pose.shape[0],
                                               embedding_dim=3,
                                               sparse=True)
            pose_codebook_trans = torch.nn.Embedding(num_embeddings=self.params_pose.shape[0],
                                               embedding_dim=3,
                                               sparse=True)
            pose_codebook_rot.weight = torch.nn.Parameter(self.params_pose[:, :3].contiguous())
            pose_codebook_trans.weight = torch.nn.Parameter(self.params_pose[:, 3:6].contiguous())
            self.optim_pose_rot = torch.optim.SparseAdam(params=pose_codebook_rot.parameters(), lr=0.005 * self.lr_scale_pose)
            self.optim_pose_trans = torch.optim.SparseAdam(params=pose_codebook_trans.parameters(), lr=0.0025 * self.lr_scale_pose)
            if self.fix_id and not self.fine_tune_id:
                self.scale_param.requires_grad = False
            self.optim_scale = torch.optim.Adam(params=[self.scale_param], lr=0.005 * self.lr_scale_pose)
        self.pose_codebook_rot = pose_codebook_rot
        self.pose_codebook_trans = pose_codebook_trans


    def update_lrs(self, epoch : int, n_epochs, epoch_mult):
        if not self.fine_tune_id:
            if epoch == int(32 * epoch_mult * len(self.expressions)):
                for param_group in self.optim_pose_rot.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_trans.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_scale.param_groups:
                    param_group["lr"] *= 1 / 2
            if epoch == int(150 * epoch_mult * len(self.expressions)):
                for param_group in self.opt.param_groups:
                    param_group["lr"] = 0.0005*self.lr_scale
                for param_group in self.opt_expr.param_groups:
                    param_group["lr"] = 0.0005*self.lr_scale
                for param_group in self.opt_sh.param_groups:
                    param_group["lr"] *= 1 / 2

            if epoch == int(200 * epoch_mult * len(self.expressions)):
                for param_group in self.opt.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_expr.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_rot.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_trans.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_scale.param_groups:
                    param_group["lr"] *= 1 / 2
            if epoch == int(225 * epoch_mult * len(self.expressions)):
                for param_group in self.opt.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_expr.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_sh.param_groups:
                    param_group["lr"] *= 1 / 2
        else:
            if epoch == n_epochs // 2:
                for param_group in self.opt.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_expr.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_sh.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_rot.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_trans.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_scale.param_groups:
                    param_group["lr"] *= 1 / 2
            if epoch == int(n_epochs * 3/4):
                for param_group in self.opt.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_expr.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.opt_sh.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_rot.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_pose_trans.param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optim_scale.param_groups:
                    param_group["lr"] *= 1 / 2


    def step(self):
        self.opt.step()
        self.opt_expr.step()
        self.opt_sh.step()
        self.optim_pose_rot.step()
        self.optim_pose_trans.step()
        self.optim_scale.step()

        self.opt.zero_grad()
        self.opt_expr.zero_grad()
        self.opt_sh.zero_grad()
        self.optim_pose_rot.zero_grad()
        self.optim_pose_trans.zero_grad()
        self.optim_scale.zero_grad()

    def compute_landmark_loss(self,
                              loss_dict,
                              epoch,
                              epoch_mult,
                              current_expression,
                              rnd_idx,
                              can_anchors,
                              data_manager : SimpleDataManager,
                              ):

        index_anchors = torch.from_numpy(self.ANCHOR_iBUG68_pairs[:, 0]).cuda()
        index_lms = torch.from_numpy(self.ANCHOR_iBUG68_pairs[:, 1]).cuda()

        _cond = {k: self.latent_code[k].clone() for k in self.latent_code.keys()}
        _cond['exp'] = self.expr_codebook(
            torch.tensor([current_expression], device='cuda')).unsqueeze(0)
        p_corresp_posed, search_result = search(can_anchors,
                                                _cond,
                                               self.idr.implicit_network.monoNPHM.ex_model,
                                                can_anchors.clone().unsqueeze(1).repeat(1, can_anchors.shape[1], 1, 1),
                                                multi_corresp=True)

        p_corresp_posed = p_corresp_posed.detach()

        num_inits = None
        if len(p_corresp_posed.shape) == 4:
            batch_size, num_points_root, num_inits, _ = p_corresp_posed.shape

            p_corresp_posed = p_corresp_posed.reshape(1, -1, 3)
            search_result['valid_ids'] = search_result['valid_ids'].reshape(-1)
        out = self.idr.implicit_network.monoNPHM.ex_model({'queries': p_corresp_posed, 'cond': _cond, 'anchors': can_anchors})
        preds_can = p_corresp_posed + out['offsets']
        grad_inv = jac(self.idr.implicit_network.monoNPHM.ex_model, {'queries': p_corresp_posed,
                                                           'cond': _cond,
                                                           'anchors': can_anchors}).inverse()
        correction = preds_can - preds_can.detach()
        correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(), correction)
        # trick for implicit diff with autodiff:
        # xc = xc_opt + 0 and xc' = correction'
        x_posed = p_corresp_posed + correction
        self.anchors_posed[current_expression] = x_posed.detach().cpu().squeeze().numpy()
        _pose_params_rot = self.pose_codebook_rot(torch.tensor([current_expression], device='cuda'))
        _pose_params_trans = self.pose_codebook_trans(torch.tensor([current_expression], device='cuda'))
        x_posed = ((x_posed - _pose_params_trans) / self.scale_param) @ so3_exp_map(_pose_params_rot).squeeze()

        if num_inits is not None:
            index_anchors = index_anchors.repeat(num_inits)  # torch.repeat(index_anchors, num_inits, dim=0)

        points2d = project_points_torch(x_posed.squeeze()[index_anchors, :],
                                        torch.from_numpy(data_manager.intrinsics[current_expression][rnd_idx]).to(
                                            x_posed.device).float(),
                                        torch.from_numpy(data_manager.c2ws[current_expression][rnd_idx]).to(x_posed.device).float())

        points2d_gt = torch.from_numpy(data_manager.lms[current_expression][rnd_idx]).float().to(x_posed.device)[index_lms, :]


        if num_inits is not None and points2d_gt is not None:
            points2d_gt = points2d_gt.repeat(num_inits, 1)

        # lm loss 2d
        points2d[:, 0] = points2d[:, 0] / data_manager.ws[current_expression][rnd_idx]
        points2d[:, 1] = points2d[:, 1] / data_manager.hs[current_expression][rnd_idx]
        points2d_gt[:, 0] = points2d_gt[:, 0] / data_manager.ws[current_expression][rnd_idx]
        points2d_gt[:, 1] = points2d_gt[:, 1] / data_manager.hs[current_expression][rnd_idx]
        valid_anchors = search_result['valid_ids'].squeeze()[index_anchors]
        points2d = points2d[valid_anchors, :]
        points2d_gt = points2d_gt.squeeze()[valid_anchors, :]
        _anchor_loss_scale = self.anchor_loss_scale.clone()
        if num_inits is not None:
            _anchor_loss_scale = _anchor_loss_scale.repeat(
                num_inits)
        anchor_loss_scale_valid = _anchor_loss_scale[valid_anchors].unsqueeze(-1)

        loss_backwarp = (anchor_loss_scale_valid * (points2d[:, :2] - points2d_gt[:, :2])).square().mean()



        if self.fine_tune_id:
            loss_dict['loss'] += loss_backwarp * 5
        else:
            if self.loss_function.rgb_weight <= 0:
                loss_dict['loss'] += loss_backwarp * 20

            else:
                if epoch < 25 * len(self.expressions) * epoch_mult:
                    loss_dict['loss'] += loss_backwarp * 100

                elif epoch < 50 * len(self.expressions) * epoch_mult:
                    loss_dict['loss'] += loss_backwarp * 50

                elif epoch < 100 * len(self.expressions) * epoch_mult:
                    loss_dict['loss'] += loss_backwarp * 20

                else:
                    loss_dict['loss'] += loss_backwarp * 10

        return loss_backwarp

    def compute_smoothness_loss(self,
                                loss_dict,
                                current_expression,
                                ):
        # smoothness constraints
        n_terms_smooth = 0
        if current_expression > 0:
            reg_loss_smooth1 = (torch.norm(
                self.expr_codebook(torch.tensor([current_expression], device='cuda')) -
                self.expr_codebook(torch.tensor([current_expression - 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth1_pose_rot = (torch.norm(
                self.pose_codebook_rot(torch.tensor([current_expression], device='cuda')) -
                self.pose_codebook_rot(torch.tensor([current_expression - 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth1_pose_trans = (torch.norm(
                self.pose_codebook_trans(torch.tensor([current_expression], device='cuda')) -
                self.pose_codebook_trans(torch.tensor([current_expression - 1], device='cuda')),
                dim=-1) ** 2).mean()

            n_terms_smooth += 1
        else:
            reg_loss_smooth1 = torch.zeros_like(loss_dict['loss'])
            reg_loss_smooth1_pose_rot = torch.zeros_like(loss_dict['loss'])
            reg_loss_smooth1_pose_trans = torch.zeros_like(loss_dict['loss'])
        if current_expression < len(self.expressions) - 1:
            reg_loss_smooth2 = (torch.norm(
                self.expr_codebook(torch.tensor([current_expression], device='cuda')) -
                self.expr_codebook(torch.tensor([current_expression + 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth2_pose_rot = (torch.norm(
                self.pose_codebook_rot(torch.tensor([current_expression], device='cuda')) -
                self.pose_codebook_rot(torch.tensor([current_expression + 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth2_pose_trans = (torch.norm(
                self.pose_codebook_trans(torch.tensor([current_expression], device='cuda')) -
                self.pose_codebook_trans(torch.tensor([current_expression + 1], device='cuda')),
                dim=-1) ** 2).mean()
            n_terms_smooth += 1
        else:
            reg_loss_smooth2 = torch.zeros_like(loss_dict['loss'])
            reg_loss_smooth2_pose_rot = torch.zeros_like(loss_dict['loss'])
            reg_loss_smooth2_pose_trans = torch.zeros_like(loss_dict['loss'])

        reg_smooth = torch.zeros_like(loss_dict['loss'])
        reg_smooth_pose = torch.zeros_like(loss_dict['loss'])
        if n_terms_smooth > 0:
            reg_smooth = (reg_loss_smooth1 + reg_loss_smooth2) / n_terms_smooth
            reg_smooth_pose_rot = (reg_loss_smooth1_pose_rot + reg_loss_smooth2_pose_rot) / n_terms_smooth
            reg_smooth_pose_trans = (reg_loss_smooth1_pose_trans + reg_loss_smooth2_pose_trans) / n_terms_smooth
            reg_smooth_pose = reg_smooth_pose_rot + reg_smooth_pose_trans * 10

        return reg_smooth, reg_smooth_pose


    def render_progress(self, data_manager : SimpleDataManager,
                        variance, epoch,
                        ):
        can_anchors = self.idr.implicit_network.monoNPHM.id_model.get_anchors(self.latent_code['geo'])

        can_anchors[:, 18, 1] += 0.005
        can_anchors[:, 19, 1] += 0.005
        can_anchors[:, 20, 1] -= 0.01
        can_anchors[:, 21, 1] -= 0.01
        can_anchors[:, 48, 1] -= 0.005
        can_anchors[:, 49, 1] -= 0.005
        can_anchors[:, 50, 1] += 0.005
        can_anchors[:, 51, 1] += 0.005

        index_anchors = torch.from_numpy(self.ANCHOR_iBUG68_pairs[:, 0]).cuda()


        for current_expression in range(len(self.expressions)):
            variances = [0.6]
            # with torch.no_grad():
            _cond = {k: self.latent_code[k].clone() for k in
                     self.latent_code.keys()}
            _cond['exp'] = self.expr_codebook(
                torch.tensor([current_expression], device='cuda')).unsqueeze(0).clone()
            p_corresp_posed, search_result = search(can_anchors,
                                                    _cond,
                                                    self.idr.implicit_network.monoNPHM.ex_model,
                                                    can_anchors.clone().unsqueeze(1).repeat(1, can_anchors.shape[
                                                        1], 1,
                                                                                            1),
                                                    multi_corresp=False)

            p_corresp_posed = p_corresp_posed.detach()
            valid_anchors = search_result['valid_ids'].squeeze()[index_anchors]

            _cond = {k: self.latent_code[k].clone() for k in
                     self.latent_code.keys()}
            _cond['exp'] = self.expr_codebook(
                torch.tensor([current_expression], device='cuda')).unsqueeze(0).clone()

            out = self.idr.implicit_network.monoNPHM.ex_model(
                {'queries': p_corresp_posed, 'cond': _cond,
                 'anchors': can_anchors})
            preds_can = p_corresp_posed + out['offsets']
            grad_inv = jac(self.idr.implicit_network.monoNPHM.ex_model, {'queries': p_corresp_posed,
                                                               'cond': _cond,
                                                               'anchors': can_anchors}).inverse()
            correction = preds_can - preds_can.detach()
            correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(), correction)
            # trick for implicit diff with autodiff:
            # xc = xc_opt + 0 and xc' = correction'
            x_posed = p_corresp_posed + correction

            _pose_params_rot = self.pose_codebook_rot(torch.tensor([current_expression], device='cuda'))
            _pose_params_trans = self.pose_codebook_trans(torch.tensor([current_expression], device='cuda'))

            x_posed = ((x_posed - _pose_params_trans) / self.scale_param) @ so3_exp_map(
                _pose_params_rot).squeeze()

            index_anchors = torch.from_numpy(self.ANCHOR_iBUG68_pairs[:, 0]).cuda()
            index_lms = torch.from_numpy(self.ANCHOR_iBUG68_pairs[:, 1]).cuda()

            for i in range(min(1, data_manager.num_views)):  # num_views):
                for v, _variance in enumerate(variances):

                    points2d = project_points_torch(x_posed.squeeze()[index_anchors, :],
                                                    torch.from_numpy(
                                                        data_manager.intrinsics[current_expression][i]).to(
                                                        x_posed.device).float(),
                                                    torch.from_numpy(data_manager.c2ws[current_expression][i]).to(
                                                        x_posed.device).float())
                    points2d_gt = torch.from_numpy(data_manager.lms[current_expression][i]).float().to(x_posed.device)[
                        index_lms]


                    points2d_px = points2d.detach().clone()
                    if points2d_gt is not None:
                        points2d_gt_px = points2d_gt.detach().clone()
                else:
                    points2d_gt = False

                # i = rnd_idx # cam index
                _full_in_dict = {k: v[current_expression][i].cuda() for k, v in data_manager.full_in_dict.items()}
                _pose_params_rot = self.pose_codebook_rot(torch.tensor([current_expression], device='cuda'))
                _pose_params_trans = self.pose_codebook_trans(torch.tensor([current_expression], device='cuda'))
                _expr_params = self.expr_codebook(torch.tensor([current_expression], device='cuda'))
                sh_coeffs = self.sh_coeffs
                condition = self.latent_code
                condition['exp'] = _expr_params.unsqueeze(0)
                I, O_mask, Pred_O_mask, sdf_pred, weights_sum, weights_img, depths = render_image(self.idr,
                                                                                                         current_expression,
                                                                                                         condition,
                                                                                                         sh_coeffs,
                                                                                                         _full_in_dict,
                                                                                                         w=data_manager.ws[
                                                                                                             current_expression][
                                                                                                             i],
                                                                                                         h=data_manager.hs[
                                                                                                             current_expression][
                                                                                                             i],
                                                                                                         n_batch_points=15000,
                                                                                                         scale_uniformly=True,
                                                                                                         variance=variance,
                                                                                                         pose_params=[
                                                                                                             _pose_params_rot,
                                                                                                             _pose_params_trans,
                                                                                                             self.scale_param],
                                                                                                         use_sh=True)  # not fix_id)
                I = ((torch.from_numpy(np.array(I)) / 255) - 0.5) * 2
                h, w, _ = I.shape
                I = I.reshape(-1, 3)
                I = I @ self.colorA.detach().cpu() + self.colorb.detach().cpu()
                I = I.reshape(h, w, 3)
                I = (I + 1) / 2 * 255
                I = I.numpy().astype(np.uint8)
                I = Image.fromarray(I)

                gt_im = np.array(data_manager.rgbs[current_expression][0])
                gt_im = np.reshape(gt_im, (data_manager.ws[current_expression][0],
                                           data_manager.hs[current_expression][0], 3))
                gt_im = ((gt_im + 1) / 2 * 255).astype(np.uint8)
                _gt_im = gt_im.copy()
                if points2d_gt is not None:
                    for anchor_idx in range(points2d_px.shape[0]):
                        color_anchor = (100, 0, 0)
                        color_lm = (0, 200, 200)
                        color_line = (100, 225, 100)
                        if not valid_anchors[anchor_idx]:
                            color_anchor = (255 // 2, 0, 0)
                            color_lm = (0, 200//2, 200 // 2)
                            color_line = (100 // 2, 225 // 2, 100 // 2)
                        gt_im = cv2.line(gt_im, (
                            int(points2d_px[anchor_idx][0].item()),
                            int(points2d_px[anchor_idx][1].item())), (
                                             int(points2d_gt_px[anchor_idx][0].item()),
                                             int(points2d_gt_px[anchor_idx][1].item())), color=color_line,
                                         thickness=1)
                        gt_im = cv2.circle(gt_im, (
                            int(points2d_px[anchor_idx][0].item()),
                            int(points2d_px[anchor_idx][1].item())),
                                           radius=1, color=color_anchor, thickness=-1)
                        gt_im = cv2.circle(gt_im, (
                            int(points2d_gt_px[anchor_idx][0].item()),
                            int(points2d_gt_px[anchor_idx][1].item())),
                                           radius=1, color=color_lm, thickness=-1)
                composed_image = np.concatenate([gt_im,
                                                 np.array(I),
                                                 weights_img,
                                                 ], axis=1)

                # ---------------------------------------------------
                ############## Error Plots #########################
                # ---------------------------------------------------
                I = (np.array(I) / 255 - 0.5) * 2
                gt_I = (np.array(_gt_im) / 255 - 0.5) * 2

                facer_mask = _full_in_dict['object_mask'].detach().cpu().squeeze().reshape(I.shape[0],
                                                                                           I.shape[1]).numpy()
                rgb_loss_mask = np.logical_not(
                    (facer_mask == 3) | (facer_mask == 0))
                foreground_mask = rgb_loss_mask

                rgb_error_img = np.mean(np.abs(I - gt_I), axis=-1)
                rgb_error_img[~rgb_loss_mask] = 0
                mm_mask = np.array(data_manager.mms[current_expression][i]) != 0
                rgb_error_img[mm_mask] /= 25

                O_mask_torch = torch.from_numpy(np.array(foreground_mask)).float()
                O_mask_pred = torch.from_numpy(weights_sum).float().clamp(0, 1)

                # TODO
                O_mask_pred[torch.isnan(O_mask_pred)] = 0
                mask_error_img = F.binary_cross_entropy(O_mask_pred, O_mask_torch, reduction='none')

                cmap = plt.get_cmap('turbo')
                rgb_error_img = cmap(np.clip(rgb_error_img * self.loss_function.rgb_weight / 30 + 0.05, 0, 1),
                                     bytes=True)[..., :3]
                mask_error_img = cmap(np.clip(mask_error_img * self.loss_function.mask_weight / 30 + 0.05, 0, 1),
                                      bytes=True)[..., :3]

                cmap_depth = plt.get_cmap('viridis')

                depths = cmap_depth(depths[..., 0] / 255, bytes=True)[..., :3]


                I_err_composed = np.concatenate([rgb_error_img, mask_error_img], axis=1)


                I_err_composed = np.concatenate([I_err_composed, depths], axis=1)


                I_composed = np.concatenate([composed_image, I_err_composed], axis=0)
                self.progress_images[current_expression][i].append(I_composed)
                I_composed = Image.fromarray(I_composed)

                I_composed.save(f'{self.progress_dir}/epoch{epoch:04d}_{current_expression}_view{i}_variance{v:05d}.png')


    def rec_and_save(self, progress_interval):
        rec_volume_cfg = self.rec_conf
        grid_points = create_grid_points_from_bounds(rec_volume_cfg['min'],
                                                     rec_volume_cfg['max'],
                                                     rec_volume_cfg['res'])
        grid_points = torch.from_numpy(grid_points).cuda().float()
        grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).cuda()

        for expression, actual_expression in enumerate(self.expressions):
            condition = self.latent_code
            condition.update(
                {'exp': self.expr_codebook(torch.tensor([expression], device='cuda')).unsqueeze(0)})
            logits = get_logits(self.idr.implicit_network.monoNPHM, condition, grid_points, nbatch_points=40000)
            mesh = mesh_from_logits(logits.copy(), rec_volume_cfg['min'], rec_volume_cfg['max'], rec_volume_cfg['res'])

            vertex_color = get_vertex_color(self.idr.implicit_network.monoNPHM,
                                            encoding=condition,
                                            vertices=torch.from_numpy(mesh.vertices).float().unsqueeze(0).cuda(),
                                            nbatch_points=40000,
                                            uniform_scaling=True,
                                            )
            vertex_color = ((vertex_color / 255) - 0.5) * 2
            if self.colorA is not None:
                vertex_color = vertex_color @ self.colorA.detach().cpu().numpy() + self.colorb.detach().cpu().numpy()
            vertex_color = ((vertex_color + 1) / 2 * 255).astype(np.uint8)
            mesh.visual.vertex_colors = vertex_color
            if self.fine_tune_id:
                os.makedirs(self.exp_dir + f'/{actual_expression:05d}/', exist_ok=True)
                mesh.export(self.exp_dir + f'/{actual_expression:05d}/mesh.ply')
            else:
                mesh.export(self.exp_dir + 'mesh.ply'.format(self.subject, actual_expression))
            torch.cuda.empty_cache()

        #if progress_interval > 0:
        #    for e, expression in enumerate(self.expressions):
        #        for i in range(min(1, self.num_views)):  # num_views):
        #            if len(self.progress_images[e][i]) > 0:
        #                mediapy.write_video(self.exp_dir + f'progress_video.mp4',
        #                                    self.progress_images[e][i], fps=1)

        # save reconstructed latent codes:
        np.save(f'{self.exp_dir}/z_geo.npy', self.latent_code['geo'].detach().cpu().numpy())
        np.save(f'{self.exp_dir}/z_app.npy', self.latent_code['app'].detach().cpu().numpy())
        if self.colorA is not None:
            np.save(f'{self.exp_dir}/colorA.npy', self.colorA.detach().cpu().numpy())
            np.save(f'{self.exp_dir}/colorb.npy', self.colorb.detach().cpu().numpy())
        for e, expression in enumerate(self.expressions):
            if self.fine_tune_id:
                os.makedirs(f'{self.exp_dir}/{expression:05d}/', exist_ok=True)
                np.save(f'{self.exp_dir}/{expression:05d}/z_exp.npy', self.expr_codebook.weight.data.detach().cpu().numpy())
            else:
                np.save(f'{self.exp_dir}/z_exp.npy', self.expr_codebook.weight.data.detach().cpu().numpy())

        if self.colorA is not None:
            colorA, colorb = self.colorA.detach().cpu().numpy(), self.colorb.detach().cpu().numpy()

        if len(self.expressions) > 1:
            for e, expression in enumerate(self.expressions):
                _pose_params_rot = self.pose_codebook_rot(torch.tensor([e], device='cuda'))
                _pose_params_trans = self.pose_codebook_trans(torch.tensor([e], device='cuda'))

                np.save(f'{self.exp_dir}/{expression:05d}/scale.npy', self.scale_param.detach().cpu().numpy())
                np.save(f'{self.exp_dir}/{expression:05d}/trans.npy', _pose_params_trans.detach().cpu().numpy())
                np.save(f'{self.exp_dir}/{expression:05d}/rot.npy', so3_exp_map(_pose_params_rot).detach().cpu().numpy())
            params_pose = None  # [so3_exp_map(params_pose[0]).detach().cpu().numpy(), params_pose[1].detach().cpu().numpy(), params_pose[2].detach().cpu().numpy()]
        else:
            _pose_params_rot = self.pose_codebook_rot(torch.tensor([0], device='cuda'))
            _pose_params_trans = self.pose_codebook_trans(torch.tensor([0], device='cuda'))

            np.save(f'{self.exp_dir}/scale.npy', self.scale_param.detach().cpu().numpy())
            np.save(f'{self.exp_dir}/trans.npy', _pose_params_trans.detach().cpu().numpy())
            np.save(f'{self.exp_dir}/rot.npy', so3_exp_map(_pose_params_rot).detach().cpu().numpy())
            params_pose = [so3_exp_map(_pose_params_trans).detach().cpu().numpy(),
                           _pose_params_rot.detach().cpu().numpy(),
                           self.scale_param.detach().cpu().numpy()]
        np.save(f'{self.exp_dir}/sh_coeffs.npy', self.sh_coeffs.detach().cpu().numpy())
        return





def track(net, cfg, subject, expressions, out_dir, out_dir_stage1=None,  fix_id=False,
        seq_name=None,   lr_scale = 1, fine_tune_id = False,
          intrinsics_provided : bool = True,
          downsampling_factor : float = 1/6,
          ):

    tracker = Tracker(subject, expressions, cfg, net, out_dir, lr_scale, fix_id, fine_tune_id=fine_tune_id,
                      out_dir_stage1=out_dir_stage1,)
    tracker.ANCHOR_iBUG68_pairs = ANCHOR_iBUG68_pairs_65
    expressions = tracker.expressions
    data_manager = SimpleDataManager(expressions, seq_name, cfg,
                                     intrinsics_provided=intrinsics_provided,
                                     downsampling_factor=downsampling_factor)


    progress_interval = 5000*len(expressions)
    if fine_tune_id:
        progress_interval = 5000*len(expressions)


    epoch_mult = 1
    if fix_id:
        if fine_tune_id:
            epoch_mult = 0.25
        else:
            if len(expressions) == 1 and expressions[0] > 0:
                if data_manager.change[expressions[0]-1] > 2*data_manager.mean_change:
                    epoch_mult = 0.5
                elif data_manager.change[expressions[0]-1] > 0.5*data_manager.mean_change:
                    epoch_mult = 0.25
                else:
                    epoch_mult = 0.1

    print(f'USING EPOCH MULT OF: {epoch_mult}')

    n_epochs = int(cfg['opt']['n_epochs']*epoch_mult*len(expressions))


    epoch = 0

    while True:
        if epoch >= n_epochs:
            break

        tracker.update_lrs(epoch, n_epochs, epoch_mult)


        current_expression, rnd_idx, in_dict = data_manager.get_random_input()

        # sample random NeuS-variance during optimization
        if fix_id or fine_tune_id:
            variance = min(0.5 + ((epoch/len(expressions)) // (50*epoch_mult) * 0.15), 0.8)
        else:
            variance = min(0.3 + ((epoch/len(expressions)) // (25*epoch_mult) * 0.15), 0.8)

        # diff. implicit-surface rendering
        _pose_params_rot = tracker.pose_codebook_rot(torch.tensor([current_expression], device='cuda'))
        _pose_params_trans = tracker.pose_codebook_trans(torch.tensor([current_expression], device='cuda'))
        cond_exp = tracker.expr_codebook(torch.tensor([current_expression], device='cuda'))
        condition = {
            'geo': tracker.latent_code['geo'],
            'app': tracker.latent_code['app'],
            'exp': cond_exp.unsqueeze(1),
        }
        #TODO: build condition
        out_dict = tracker.idr(in_dict, condition, skip_render=False, neus_variance=variance,
                       pose_params=[_pose_params_rot, _pose_params_trans, tracker.scale_param],
                               sh_coeffs=tracker.sh_coeffs,
                               w2c=torch.from_numpy(data_manager.c2ws[current_expression][0]).float().cuda() # TODO: move into datamanger
                               )

        # obtain latent regularization
        reg_loss = tracker.idr.implicit_network.monoNPHM.id_model.get_reg_loss(tracker.latent_code)
        reg_loss_expr = (torch.norm(tracker.expr_codebook(torch.tensor([current_expression], device='cuda')), dim=-1) ** 2).mean()
        out_dict['reg_loss'] = reg_loss
        out_dict['reg_loss_expr'] = reg_loss_expr



        out_dict['rgb_values'] = out_dict['rgb_values'] @ tracker.colorA + tracker.colorb
        # compute loss
        loss_dict = tracker.loss_function(out_dict, in_dict)


        can_anchors = tracker.idr.implicit_network.monoNPHM.id_model.get_anchors(tracker.latent_code['geo'])
        can_anchors[:, 18, 1] += 0.005
        can_anchors[:, 19, 1] += 0.005
        can_anchors[:, 20, 1] -= 0.01
        can_anchors[:, 21, 1] -= 0.01
        can_anchors[:, 48, 1] -= 0.005
        can_anchors[:, 49, 1] -= 0.005
        can_anchors[:, 50, 1] += 0.005
        can_anchors[:, 51, 1] += 0.005

        loss_backwarp = tracker.compute_landmark_loss(loss_dict, epoch, epoch_mult, current_expression, rnd_idx, can_anchors, data_manager)

        reg_smooth, reg_smooth_pose = tracker.compute_smoothness_loss(loss_dict, current_expression)
        loss_dict['smoothness_expr'] = reg_smooth*100
        loss_dict['smoothness_pose'] = reg_smooth_pose*500

        if epoch > n_epochs//2:
            loss_dict['loss'] += reg_smooth * 500*2
        else:
            loss_dict['loss'] += reg_smooth * 1000*2
        if epoch > n_epochs//2:
            loss_dict['loss'] += reg_smooth_pose * 5000*5
        else:
            loss_dict['loss'] += reg_smooth_pose * 10000*5


        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_([tracker.latent_code['geo']], 0.1)
        torch.nn.utils.clip_grad_norm_([tracker.latent_code['app']], 0.1)
        tracker.step()


        print_str = f'Epoch: {epoch}, '
        for k in loss_dict.keys():
            print_str += f'{k}: {loss_dict[k].item():3.5f}, '

        print_str += f' LM loss: {loss_backwarp.item()}'

        print(print_str)


        # render current revconstruction state for debugging purposes
        if progress_interval > 0 and epoch % int(progress_interval*epoch_mult) == 0 and epoch > 0 or (epoch == n_epochs-1):
            tracker.render_progress(data_manager, variance, epoch)

        epoch += 1

        del loss_backwarp
        loss_dict.clear()
        in_dict.clear()
        out_dict.clear()

    torch.cuda.empty_cache()
    tracker.rec_and_save( progress_interval)

    return