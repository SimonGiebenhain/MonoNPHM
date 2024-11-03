import dreifus.pyvista
import trimesh
import numpy as np
import networkx as nx
import torch

import pyvista as pv
from pyvista import global_theme
import os
import time
import math
from PIL import Image
import cv2


from pytorch3d.ops import knn_points


from dreifus.render import project
from mononphm.utils_3d.mesh_operations import laplace_regularizer_const, enlarge_region
from mononphm.utils_3d.render import render, project_torch
from mononphm import env_paths
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles



def start_xvfb(wait=3, window_size=None, display: int = 99, screen: int = 0):
    """
    Adaption of pyvista's start_xvfb() because in multi-user server environments, DISPLAY=:99.0 might already be taken.
    This would give a "bad X server connection" error.
    The solution is, to simply use a different display number.
    """

    XVFB_INSTALL_NOTES = """Please install Xvfb with:
                                    Debian
                            $ sudo apt install libgl1-mesa-glx xvfb

                            CentOS / RHL
                            $ sudo yum install libgl1-mesa-glx xvfb

                            """

    pv.start_xvfb()

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    if os.system('which Xvfb > /dev/null'):
        raise OSError(XVFB_INSTALL_NOTES)

    # use current default window size
    if window_size is None:
        window_size = global_theme.window_size
    window_size_parm = f'{window_size[0]:d}x{window_size[1]:d}x24'
    display_num = f':{display}'
    os.system(f'Xvfb {display_num} -screen {screen} {window_size_parm} > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = display_num
    if wait:
        time.sleep(wait)



def get_ridid_alignment(mesh, pointcloud, normals, is_flame=True):
    icp = ICP(mesh, pointcloud, normals, is_flame=is_flame)

    params_opt = {'R': torch.zeros(1, 3).float().cuda(),
                  't': torch.zeros(1, 3).float().cuda(),
                  's': torch.ones(1).float().cuda(),
                  }
    for k in params_opt.keys():
        params_opt[k].requires_grad = True

    opt = torch.optim.Adam(params=[params_opt['R'], params_opt['t'], params_opt['s']], lr=0.001) #2e-3)

    step_mult = 1/10
    N_ITER = int(2500*step_mult)
    for jj in range(N_ITER):
        opt.zero_grad()
        loss = icp.residuals(params_opt)
        loss.backward()
        opt.step()
        if jj == int(500*step_mult):
            for g in opt.param_groups:
                g['lr'] /= 2
        if jj == int(2000*step_mult):
            for g in opt.param_groups:
                g['lr'] /= 5

        if jj % 100 == 0:
            print(params_opt['R'].detach().norm().item(), params_opt['t'].detach().norm().item())

    #return {k: params_opt[k].detach().cpu().numpy() for k in params_opt.keys()}
    return {
            'R': euler_angles_to_matrix(params_opt['R'], convention='XYZ').squeeze().detach().cpu().numpy(),
            't': params_opt['t'].detach().cpu().numpy(),
            's': params_opt['s'].detach().cpu().numpy(),
            }

class ICP:
    def __init__(self, mesh,
                 pointcloud,
                 normalcloud,
                 #target_landmarks3d,
                 normal_threshold: float = 0.2,
                 distance_threshold: float = 0.01,
                 writer=None,
                 exp_dir=None,
                 supress_pv : bool = False,
                 is_flame = True

                 ):

        self.frames = None #TODO


        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.SUPRESS_PV = supress_pv
        self.normal_threshold = normal_threshold
        self.distance_threshold = distance_threshold

        self.mask = None

        self.n_mesh = torch.from_numpy(mesh.vertex_normals).float().cuda()
        self.v_mesh = torch.from_numpy(mesh.vertices).float().cuda()
        #self.lm_target = torch.from_numpy(target_landmarks3d).cuda()
        self.v_target = torch.from_numpy(pointcloud).float().cuda()
        self.n_target = torch.from_numpy(normalcloud).float().cuda()

        self.writer = writer
        self.exp_dir = exp_dir

        self.iter = 0
        self.step = 0

        if is_flame:
            self.set_up_vertex_mask()


    def set_up_vertex_mask(self):
        m_color = trimesh.load(env_paths.ASSETS + 'test_rigid.ply')
        #m_color.show()
        colors = m_color.visual.vertex_colors

        inner_lip_mask = np.logical_and(np.logical_and(colors[:, 0] == 0, colors[:, 1] == 0), colors[:, 2] == 255)
        inner_lip_mask = np.logical_and(inner_lip_mask, m_color.vertices[:, 2] < 0.0)
        inner_lip_idx = np.where(inner_lip_mask)[0]

        # enlarge region by including k-hop neighborhood
        g = nx.from_edgelist(m_color.edges_unique)
        one_ring = [list(g[i].keys()) for i in range(len(m_color.vertices))]
        enlarged_idx = inner_lip_idx.copy()
        for i in range(3):
            enlarged_idx = enlarge_region(enlarged_idx.copy(), one_ring).copy()
        template_mask = np.ones([m_color.vertices.shape[0]], dtype=np.bool)
        template_mask[enlarged_idx] = 0

        # mask out baback of the head/hair region
        hair_region = np.logical_and(np.logical_and(colors[:, 0] == 255, colors[:, 1] == 0), colors[:, 2] == 0)
        template_mask[hair_region] = 0

        template_mask[~np.all(colors[:, :3]==0, axis=-1)] = 0

        self.mask = torch.from_numpy(template_mask).to(self.device)


    def get_corresp(self, xs):
        all_matches = []
        all_n_targets = []
        all_invalid = []
        all_scalar = []
        #self.distances, self.indices_matches = self.index.search(xs.float().contiguous(), 1, self.distances, self.indices_matches)
        #matches = self.v_target[self.indices_matches.squeeze(), :]
        knns = knn_points(xs.float().unsqueeze(0).contiguous(), self.v_target.unsqueeze(0).float(), return_nn=False)
        matches = self.v_target[knns.idx.squeeze(), :]
        self.indices_matches = knns.idx.squeeze()
        self.distances = knns.dists.squeeze()
        if self.writer is not None:
            correct_matches = self.indices_matches[i].squeeze() == torch.arange(xs[i].shape[0], device=self.device)
            n_correct_mathes = torch.count_nonzero(correct_matches)
            self.writer.add_scalar("recall_before", n_correct_mathes / xs[i].shape[0], self.step)
            self.writer.add_scalar("precision_before", n_correct_mathes / xs[i].shape[0], self.step)

        n_target = self.n_target[self.indices_matches.squeeze(), :]

        scalar = (self.n_mesh * n_target).sum(-1)
        invalid_normal = scalar < self.normal_threshold #0.2
        distance = (matches - xs).norm(dim=-1)
        invalid_distance = distance > self.distance_threshold #0.005
        invalid = torch.logical_or(invalid_normal, invalid_distance)

        all_matches.append(matches)
        all_n_targets.append(n_target)
        all_invalid.append(invalid)
        all_scalar.append(scalar)
        return matches, n_target, invalid, scalar


    def get_landmark_loss(self, all_lms : torch.tensor ):
        total_loss = 0
        residuals_pip = ((all_lms - self.lm_target_pip)).abs()#**2
        residuals = residuals_pip
        # jawline
        jaw_idx = torch.arange(17, device=self.device)
        residuals[:, jaw_idx, :] *= 0.01
        #mouth
        mouth_idx = torch.arange(48, 68, device=self.device)
        residuals[:, mouth_idx, :] *= 32#2
        # eyes
        eye_idx = torch.arange(36, 48, device=self.device)
        residuals[:, eye_idx, :] *= 1280  # 8
        # pupils:
        residuals[:, -2:, :] *= 1280

        #residuals[-2:, :] *= 2  # 4
        residuals = residuals[~torch.isnan(residuals[..., 0]), :]
        total_loss += residuals.sum()
        total_loss = total_loss / all_lms.shape[0]
        return total_loss

    def get_landmark_loss_2d(self, all_lms : torch.tensor ):
        total_loss = 0

        predicted_lms_2d = project_torch(all_lms, self.w2cs, self.intrinsics)

        residuals_pip = ((predicted_lms_2d - self.lm_target_pip2d)).abs()#**2
        residuals_pip[..., 0] /= 1100
        residuals_pip[..., 1] /= 1604
        residuals = residuals_pip
        # jawline
        jaw_idx = torch.arange(17, device=self.device)
        residuals[:, jaw_idx, :] *= 1
        #mouth
        mouth_idx = torch.arange(48, 68, device=self.device)
        residuals[:, mouth_idx, :] *= 20#2
        # eyes
        eye_idx = torch.arange(36, 48, device=self.device)
        residuals[:, eye_idx, :] *= 100  # 8
        #pupils:
        residuals[:, -2:, :] *= 25000

        #residuals[-2:, :] *= 2  # 4
        residuals = residuals[~torch.isnan(residuals[..., 0]), :]
        total_loss += residuals.sum()
        total_loss = total_loss / all_lms.shape[0]
        return total_loss


    def get_geometric_loss(self, pred_xs):
        corrs, corrs_n, invalids_n, _ = self.get_corresp(pred_xs)
        loss_geom_point2point = ((pred_xs - corrs).abs()).sum(dim=-1)
        loss_geom_point2plane = ((pred_xs - corrs) * corrs_n).sum(dim=-1).abs()
        loss_geom = loss_geom_point2plane * 0.9 + loss_geom_point2point * 0.1
        loss_geom[invalids_n] = 0

        if self.mask is not None:
            loss_geom = loss_geom[self.mask]
        combined_loss_geom = loss_geom.sum() / pred_xs.shape[0]

        if False and self.step % 100 == 0:
            pl = pv.Plotter()
            if self.mask is not None:
                pl.add_points(pred_xs[self.mask].detach().cpu().numpy(), scalars=loss_geom.detach().cpu().numpy())
            else:
                pl.add_points(pred_xs.detach().cpu().numpy(), scalars=loss_geom.detach().cpu().numpy())
            pl.add_points(self.v_target.detach().cpu().numpy(), color='red')
            pl.show()

        return combined_loss_geom


    def get_smoothness_loss_vertex_space(self, pred_verts):
        N_timesteps = pred_verts.shape[0]

        smoothness = (pred_verts[1:N_timesteps, ...] - pred_verts[:N_timesteps-1]).norm(dim=-1).abs().mean()

        return {'vertex_smoothness': smoothness
                }

    def get_smoothness_loss_parameter_space(self, variables):
        N_timesteps = variables['expression'].shape[0]

        smoothness_exp = (variables['expression'][1:N_timesteps, ...] - variables['expression'][:N_timesteps-1]).norm(dim=-1).square().mean()
        smoothness_rigid = (((variables['rotation'][1:N_timesteps, ...] - variables['rotation'][:N_timesteps-1]) / 2 / math.pi) ** 2).sum(dim=-1).mean() + \
                            ((variables['translation'][1:N_timesteps, ...] - variables['translation'][:N_timesteps-1]) ** 2).sum(dim=-1).mean()
        smoothness_pose = ((variables['jaw'][1:N_timesteps, ...] - variables['jaw'][:N_timesteps-1]) ** 2).sum(dim=-1).mean()
        return {'expression': smoothness_exp,
                'rigid': smoothness_rigid, #*10,
                'jaw_pose': smoothness_pose, #*10,
                }

    def get_regularization_loss(self, variables):
        reg_shape = variables['shape'].norm() ** 2

        reg_exp = variables['expression'].norm(dim=-1).square().mean()
        reg_pose = (variables['jaw'] ** 2).sum(dim=-1).mean()
        reg_rigid = ((variables['rotation'] / 2 / math.pi) ** 2).sum(dim=-1).mean() + \
                     (variables['translation'] ** 2).sum(dim=-1).mean()


        reg_rigid += (variables['scale'].squeeze() - 1) ** 2
        return {'shape': reg_shape,
                'expression': reg_exp,
                'rigid': reg_rigid,
                'jaw_pose': reg_pose}

    def residuals(self, variables):
        self.step += 1

        #loss_landmarks = self.get_landmark_loss(x_lms)
        #loss_landmarks_2d = self.get_landmark_loss_2d(x_lms)
        pred_xs = variables['s'] * self.v_mesh @ euler_angles_to_matrix(variables['R'], convention='XYZ').squeeze().T + variables['t']
        loss_geometric = self.get_geometric_loss(pred_xs)

        #loss_reg = self.get_regularization_loss(variables)
        #loss_smooothness = self.get_smoothness_loss_vertex_space(pred_verts=xs)

        if self.writer is not None:
            self.writer.add_scalar("mean_lm_error", loss_landmarks, self.step)
            self.writer.add_scalar("mean_lm_error_2d", loss_landmarks_2d, self.step)
            self.writer.add_scalar("mean_geo_error", loss_geometric, self.step)


        self.iter += 1



        logging_string = 'Step: {:05d}, ' \
                         'LossGeom: {:2.6f}, ' \
                         .format(self.iter,
                                                   #loss_landmarks.item(),
                                                   #loss_landmarks_2d.item(),
                                                   loss_geometric.item(),
                                                   #loss_reg['shape'].item(),
                                                   #loss_reg['expression'].item()
                        )

        #logging_string += ' Smooth.Exp: {:2.6f}, ' \
        #                  'Smooth.Rigid: {:2.6f}, ' \
        #                  'Smooth.JawPose: {:2.6f}'.format(loss_smooothness['expression'],
        #                                                loss_smooothness['rigid'],
        #                                                loss_smooothness['jaw_pose'])
        #logging_string += 'Smooth. : {:2.6f}'.format(loss_smooothness['vertex_smoothness'])


        if self.iter % 100 == 0:
            print(logging_string)

        # TODO expose lambdas / schedule
        lam_geom = 1
        if self.iter > 100:
            lam_geom = 0.05
        #if self.iter > 1000:
        #    loss_landmarks *= 0.1

        residuals = loss_geometric / lam_geom #loss_landmarks + loss_landmarks_2d*0.1 + \
                    #loss_reg['expression'] / 10 + \
                    #loss_reg['shape'] / 10 + \
                    #loss_reg['jaw_pose'] / 5 + \
                    #loss_reg['rigid'] / 5
        #if self.enforce_smoothness:
            #residuals += loss_smooothness['expression']/10 + \
            #             loss_smooothness['jaw_pose']/5 + \
            #             loss_smooothness['rigid']/5
        #    residuals += loss_smooothness['vertex_smoothness'] * 5000

        # update normals of current estimate

        #if self.step % 300 == 0:
        #    if self.exp_dir is not None and not self.SUPRESS_PV:
        #        self.viz(variables)

        return residuals


    def viz(self, variables, show=False, arap=None, ptp_dist=None, close_up=False, transform=False, save_render=True,
            cam_pose = None, intrinsics = None):
        preds, pred_lms = self.warp(variables)
        preds_noT, pred_lms_noT = self.warp(variables, apply_similarity_transform=transform)
        corrs, corrs_n, invalids_n, _ = self.get_corresp(preds)

        ms_source = []
        ms_target = []
        ms_warped = []
        for ii in range(len(variables['expression'])):

            target_pc = self.v_target[ii]
            target_n = self.n_target[ii]
            s = variables['scale']
            R = euler_angles_to_matrix(variables['rotation'][ii], 'XYZ').squeeze()
            t = variables['translation'][ii]
            target_pc = 1/s*(target_pc - t) @ R #transform target point cloud into FLAME space
            target_pc = target_pc.detach().cpu().numpy()
            target_n = (target_n @ R).detach().cpu().numpy()
            with torch.no_grad():

                if preds[ii].shape == self.v_target[ii].shape:
                    loss = torch.sqrt(torch.sum((preds[ii] - self.v_target[ii]) ** 2, dim=-1)).detach().cpu().numpy()

                if not self.SUPRESS_PV:
                    pl = pv.Plotter(shape=(2, 3), off_screen=not show)

                m_source = self.template[0].copy()
                m_source.vertices = 1 / s.detach().cpu().numpy() * (m_source.vertices - t.detach().cpu().numpy()) @ R.detach().cpu().numpy()  # transform target point cloud into FLAME space
                m_warped = trimesh.Trimesh(preds_noT[ii].detach().cpu().numpy(), self.template[0].faces, process=False)
                m_tmp = preds_noT[ii].detach().cpu().numpy()
                m_warped_white = pv.wrap(trimesh.Trimesh(m_tmp, self.template[0].faces, process=False))
                m_warped_white2 = pv.wrap(trimesh.Trimesh(m_tmp, self.template[0].faces, process=False))

                lms = self.lm_target_pip[ii]
                lms = 1 / s * (lms - t) @ R  # transform target point cloud into FLAME space
                lms = lms.detach().cpu().numpy()
                lms_pred = pred_lms_noT[ii].detach().cpu().numpy()

                if not self.SUPRESS_PV:
                    pl.subplot(0, 0)
                    pl.add_points(target_pc, scalars=((target_n+1)/2*255).astype(np.uint8), rgb = True)
                    self.set_cam(pl)

                    pl.subplot(1, 1)
                    pl.add_points(target_pc, scalars=((target_n+1)/2*255).astype(np.uint8), rgb = True)
                    pl.add_points(preds_noT[ii].detach().cpu().numpy(), scalars=invalids_n[ii].detach().cpu().numpy())
                    self.set_cam(pl)

                    pl.subplot(0, 2)
                    if arap is not None:
                        pl.add_mesh(m_warped_white, scalars=arap.sum(dim=1).cpu().numpy(),
                                    scalar_bar_args={'title': 'PVR'})

                    else:
                        pl.add_mesh(m_warped_white)
                    self.set_cam(pl)

                    pl.subplot(1, 0)
                    pl.add_mesh(m_source)
                    self.set_cam(pl)

                    pl.subplot(0, 1)
                    pl.add_mesh(m_warped_white2)
                    pl.add_points(lms, color='red')
                    pl.add_points(lms_pred, color='blue')
                    for k in range(lms.shape[0]):
                        pl.add_mesh(pv.Line(lms[k, :], lms_pred[k, :]))
                    self.set_cam(pl)

                    pl.subplot(1, 2)
                    if preds[ii].shape == self.v_target[ii].shape:
                        if ptp_dist is not None:
                            pl.add_mesh(m_warped, scalars=ptp_dist.detach().cpu().numpy(), scalar_bar_args={'title': 'PVE'})
                        else:
                            pl.add_mesh(m_warped, scalars=loss, scalar_bar_args={'title': 'PVE'})

                    else:
                        pl.add_mesh(m_warped)
                    pl.link_views()
                    self.set_cam(pl)
                    if self.frames is None:
                        self.frames = list(range(len(self.v_target)))
                    os.makedirs(self.exp_dir + '/progress_frame_{}/'.format(self.frames[ii]), exist_ok=True)
                    if save_render:
                        if cam_pose is None:
                            pl.show(screenshot=self.exp_dir + '/progress_frame_{}/{:05d}.png'.format(self.frames[ii], self.iter))
                        else:
                            image = dreifus.pyvista.render_from_camera(pl, cam_pose, intrinsics)[..., :3]
                            I = Image.fromarray(image)
                            I.save(self.exp_dir + '/progress_frame_{}/{:05d}.jpg'.format(self.frames[ii], self.iter))

                    pl.close()
                ms_source.append(m_source)
                ms_target.append(self.v_target[ii].detach().cpu().numpy())
                ms_warped.append(m_warped)
        return ms_source, \
               [self.v_target[ii].detach().cpu().numpy() for ii in range(len(self.v_target))], \
               ms_warped

    def viz_final_backup(self, variables, cam_pose, intrinsics, rgbs=None, off_screen=True):

        #if os.path.exists('/cluster/doriath/'):
        start_xvfb(display=96)
        preds_noT, pred_lms_noT = self.warp(variables, apply_similarity_transform=True)

        for ii in range(len(variables['expression'])):

            target_pc = self.v_target[ii].detach().cpu().numpy()
            target_n = self.n_target[ii].detach().cpu().numpy()
            with torch.no_grad():

                #pl = pv.Plotter(window_size=(int(intrinsics.cx*2)+1, int(intrinsics.cy*2)+1),
                if not off_screen:
                    pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]))
                else:
                    pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]),
                    off_screen=True,
                    lighting='none')

                m_tmp = preds_noT[ii].detach().cpu().numpy()
                m_warped_white = pv.wrap(trimesh.Trimesh(m_tmp, self.template[0].faces, process=False))

                lms = self.lm_target_pip[ii]
                lms = lms.detach().cpu().numpy()
                lms_pred = pred_lms_noT[ii].detach().cpu().numpy()


                pl.add_points(target_pc, scalars=((target_n+1)/2*255).astype(np.uint8), rgb = True)
                #pl.add_points(preds_noT[ii].detach().cpu().numpy(), scalars=invalids_n[ii].detach().cpu().numpy())
                pl.add_mesh(m_warped_white.copy())

                pl.add_points(lms_pred, color='red', point_size=10)
                pl.add_points(lms, color='orange', point_size=10)

                os.makedirs(self.exp_dir + '/result_plots/', exist_ok=True)

                #image = dreifus.pyvista.render_from_camera(pl, cam_pose, intrinsics)[..., :3]
                if off_screen:
                    image = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                else:
                    pl.show()
                    #image = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                pl.close()
                #pl = pv.Plotter(window_size=(int(intrinsics.cx * 2) + 1, int(intrinsics.cy * 2) + 1), off_screen=True, lighting='none')
                pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]), off_screen=True, lighting='none')
                pl.add_mesh(m_warped_white)
                pl.add_points(lms_pred, color='red', point_size=10)
                pl.add_points(lms, color='green', point_size=10)
                image2 = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                #image2 = dreifus.pyvista.render_from_camera(pl, cam_pose, intrinsics)[..., :3]
                composed = rgbs['222200037'][ii].copy()
                alpha_mask = image2[..., 2] != 255
                composed[alpha_mask, :] = 0.6*image2[alpha_mask, :3] + 0.4*composed[alpha_mask, :]

                lms_detected_2d = project(lms, cam_pose, intrinsics)
                gt_img = rgbs['222200037'][ii]

                for lm_index in range(lms_detected_2d.shape[0]):
                    if not np.any(np.isnan(lms_detected_2d[lm_index, :])):
                        gt_img = cv2.circle(gt_img, (int(lms_detected_2d[lm_index, 0]), int(lms_detected_2d[lm_index, 1])), radius=0, color=(0, 0, 255), thickness=3)

                if off_screen:
                    image_sidebyside = np.concatenate([image, composed, gt_img], axis=1)
                    I = Image.fromarray(image_sidebyside)
                    I.save(self.exp_dir + '/result_plots/{:05d}.jpg'.format(self.frames[ii], self.iter))

    def viz_final(self, variables, cam_pose, intrinsics, rgbs=None, off_screen=True, port_num : int = 96):

        if os.path.exists('/cluster/doriath/'):
            start_xvfb(display=port_num)
        preds_noT, pred_lms_noT = self.warp(variables, apply_similarity_transform=True)

        for ii in range(len(variables['expression'])):

            target_pc = self.v_target[ii].detach().cpu().numpy()
            target_n = self.n_target[ii].detach().cpu().numpy()
            with torch.no_grad():

                # pl = pv.Plotter(window_size=(int(intrinsics.cx*2)+1, int(intrinsics.cy*2)+1),
                if not off_screen:
                    pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]))
                else:
                    pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]),
                                    off_screen=True,
                                    lighting='none')

                m_tmp = preds_noT[ii].detach().cpu().numpy()
                m_warped_white = pv.wrap(trimesh.Trimesh(m_tmp, self.template[0].faces, process=False))

                lms = self.lm_target_pip[ii]
                lms = lms.detach().cpu().numpy()
                lms_pred = pred_lms_noT[ii].detach().cpu().numpy()

                pl.add_points(target_pc, scalars=((target_n + 1) / 2 * 255).astype(np.uint8), rgb=True)
                # pl.add_points(preds_noT[ii].detach().cpu().numpy(), scalars=invalids_n[ii].detach().cpu().numpy())
                pl.add_mesh(m_warped_white.copy())
                pl.add_points(lms_pred, color='red', point_size=10)
                pl.add_points(lms, color='orange', point_size=10)

                os.makedirs(self.exp_dir + '/result_plots/', exist_ok=True)

                # image = dreifus.pyvista.render_from_camera(pl, cam_pose, intrinsics)[..., :3]
                if off_screen:
                    image = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                else:
                    pl.show()
                    # image = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                pl.close()
                # pl = pv.Plotter(window_size=(int(intrinsics.cx * 2) + 1, int(intrinsics.cy * 2) + 1), off_screen=True, lighting='none')
                pl = pv.Plotter(window_size=(rgbs['222200037'][ii].shape[1], rgbs['222200037'][ii].shape[0]),
                                off_screen=True, lighting='none')
                pl.add_mesh(m_warped_white)
                image2 = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]
                # image2 = dreifus.pyvista.render_from_camera(pl, cam_pose, intrinsics)[..., :3]
                composed = rgbs['222200037'][ii].copy()
                alpha_mask = image2[..., 2] != 255
                composed[alpha_mask, :] = 0.6 * image2[alpha_mask, :3] + 0.4 * composed[alpha_mask, :]

                lms_detected_2d = project(lms, cam_pose, intrinsics)
                gt_img = rgbs['222200037'][ii]

                for lm_index in range(lms_detected_2d.shape[0]):
                    if not np.any(np.isnan(lms_detected_2d[lm_index, :])):
                        gt_img = cv2.circle(gt_img,
                                            (int(lms_detected_2d[lm_index, 0]), int(lms_detected_2d[lm_index, 1])),
                                            radius=0, color=(0, 0, 255), thickness=3)

                if off_screen:
                    image_sidebyside = np.concatenate([image, composed, gt_img], axis=1)
                    I = Image.fromarray(image_sidebyside)
                    I.save(self.exp_dir + '/result_plots/{:05d}.jpg'.format(self.frames[ii], self.iter))

    def viz_final_debug(self, variables, cam_pose, intrinsics, rgbs=None, off_screen=True):

        if os.path.exists('/cluster/doriath/'):
            start_xvfb(display=96)

        pl = pv.Plotter(window_size=(1000, 1000),
                        off_screen=True,
                        lighting='none')


        pl.add_points(np.random.randn(1000, 3))

        image = render(pl, pred_mesh=None, c2w_extrinsics=cam_pose, intrinsics=intrinsics)[..., :3]

        pl.close()
        I = Image.fromarray(image)
        I.save(self.exp_dir + '/result_plots_debug.jpg'.format())


    def set_cam(self, pl):
            pl.camera_position = (0, 0, 10)
            pl.camera.zoom(2)
            pl.camera.roll = 0
