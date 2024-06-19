import os.path

import torch
import trimesh


from dreifus.camera import CameraCoordinateConvention

from mononphm import env_paths

import numpy as np
from PIL import Image
from dreifus.matrix import Pose, Intrinsics
from scipy import ndimage

from mononphm.utils.renderer import m3dLookAt, back_project
from mononphm.utils.transformations import invert_similarity_transformation


WFLW_2_iBUG68 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]


def prepare_data(timestep=None,
                 seq_name=None,
                 downsample_factor=1 / 6,
                 intrinsics_provided = True,):

    # list inside "data" are for multiple views; this codebase always uses a single view though
    data = {
        'rgb': [],
        'segmentation_mask': [],
        'segmentation_mask_hq': [],
        'view_dir': [],
        'view_dir_hq': [],
        'cam_pos': [],
        'width': [],
        'width_hq': [],
        'height': [],
        'height_hq': [],
        'w2c': [],
        'intrinsics': [],
        'intrinsics_hq': [],
        'mouth_interior_mask': [],
        'landmarks_2d': [],
        'normal_map': [],
    }

    # load head pose from MICA tracking
    try:
        new_mica_pose = np.load(f'{env_paths.DATA_TRACKING}/{seq_name}/metrical_tracker/{seq_name}/checkpoint/{max(timestep - 1, 0):05d}_cam_params_opencv.npz') #TODO
    except Exception as fehler:
        new_mica_pose = np.load(f'{env_paths.DATA_TRACKING2}/s{seq_name[6:]}/checkpoint/{max(timestep - 1, 0):05d}_cam_params_opencv.npz')

    mica_pose = new_mica_pose

    if intrinsics_provided:
        # load intrinsics parameters
        calibration_folder = f'{env_paths.ASSETS}/kinect_intrinsics.txt'
        with open(calibration_folder, 'r') as fp:
            print("Loading intrinsics from JSON")
            calibration = fp.readline().split()
            fx = float(calibration[2])
            fy = float(calibration[3])
            cx = float(calibration[0])
            cy = float(calibration[1])
        # construct intrinsics matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        # load intrinsics from MICA estiamte
        K = mica_pose['K'][0, :, :]


    # create cam2world space extrinsics in OPENGL convention
    w2c_pose_mica = Pose(mica_pose['R'][0], mica_pose['t'][0], camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
    # transform to OPEN_GL convention
    w2c_pose_mica = w2c_pose_mica.invert().change_camera_coordinate_convention(new_camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL).invert()
    flame2camera_space = np.array(w2c_pose_mica)
    opengl_iden_cam = np.array(Pose(np.eye(4), camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL))
    c2w_ex = invert_similarity_transformation(flame2camera_space) @ opengl_iden_cam
    # adjust scale from FLAME to NPHM
    c2w_ex[:3, 3] *= 4

    try:
        rgb = Image.open(f'{env_paths.DATA_TRACKING}/{seq_name}/source/{timestep:05d}.png') #TODO
        facer_mask = Image.open(f'{env_paths.DATA_TRACKING}/{seq_name}/seg/{timestep}.png')
    except Exception as fehler:
        rgb = Image.open(f'{env_paths.KINECT_DATA}/{seq_name}/color/{timestep:05d}.png')
        facer_mask = Image.open(f'{env_paths.KINECT_DATA2}/{seq_name}/seg/{timestep}.png')

    rgb_og = np.array(rgb).copy()
    #rgb = Image.open(f'{env_paths.KINECT_raw}/{seq_name}/color/{timestep:05d}.png')

    matting_mask = Image.open(f'{env_paths.DATA_TRACKING}/{seq_name}/matting/{timestep:05d}.png')


    facer_mask = np.array(facer_mask)
    matting_mask = np.array(matting_mask)

    # shrink hair region where matting is not as confident to be foreground
    rgb_loss_mask = (np.logical_not( (facer_mask == 3) |(facer_mask <= 1) | (facer_mask == 18)) | (facer_mask==14))
    if np.max(facer_mask) > 14:
        rgb_loss_mask = rgb_loss_mask & ~(facer_mask == np.max(facer_mask))
    foreground_mask = rgb_loss_mask | (facer_mask == 1)  # include neck
    foreground_mask = foreground_mask & (matting_mask > 0.8*255)
    facer_mask[~foreground_mask] = 0

    # mouth interior
    mouth_interior_mask = facer_mask == 11

    rgb = np.array(rgb)

    # load landmarks and scale s.t. unit is in pixels
    #detected_lms = np.load(f'{env_paths.KINECT_data}/{seq_name}/pipnet/test.npy')[timestep, :]
    try:
        detected_lms = np.load(f'{env_paths.DATA_TRACKING}/{seq_name}/pipnet/test.npy')[timestep, :]
    except Exception as fehler:
        detected_lms = np.load(f'{env_paths.KINECT_DATA2}/{seq_name}/pipnet/test.npy')[timestep, :]
    detected_lms[:, 0] *= rgb.shape[1]
    detected_lms[:, 1] *= rgb.shape[0]

    if intrinsics_provided:
        avg_landmark = np.mean(detected_lms, axis=0).astype(int)
        # fixed square crop of size: 800 pixels, unless it would go out of bounds
        box_size_x = 2 * min(400, min(avg_landmark[0], rgb.shape[1] - avg_landmark[0] - 1))
        box_size_y = 2 * min(400, min(avg_landmark[1], rgb.shape[0] - avg_landmark[1] - 1))
        box_size = min(box_size_x, box_size_y)

        box_width = box_size
        box_height = box_size
        left_start = avg_landmark[0] - box_width // 2
        top_start = avg_landmark[1] - box_height // 2
    else:

        box_width = rgb.shape[1]
        box_height = rgb.shape[0]
        left_start = 0
        top_start = 0

    # move detected landmarks into crop
    detected_lms[:, 0] -= left_start
    detected_lms[:, 1] -= top_start

    # crop image and masks
    rgb = rgb[top_start:top_start + box_height, left_start:left_start + box_width]
    mouth_interior_mask = mouth_interior_mask[top_start:top_start + box_height, left_start:left_start + box_width]
    facer_mask = facer_mask[top_start:top_start + box_height, left_start:left_start + box_width]




    # scale image/rendering size
    rend_size_hq = (int((box_height)), int((box_width)))
    rend_size = (int(downsample_factor*(box_height)), int(downsample_factor*(box_width)))
    rgb_hq = Image.fromarray(rgb).resize((rend_size_hq[1], rend_size_hq[0]))
    rgb = Image.fromarray(rgb).resize((rend_size[1], rend_size[0]))
    mouth_interior_mask = Image.fromarray(mouth_interior_mask).resize((rend_size[1], rend_size[0]), Image.NEAREST)
    facer_mask_hq = Image.fromarray(facer_mask).resize((rend_size_hq[1], rend_size_hq[0]), Image.NEAREST)
    facer_mask = Image.fromarray(facer_mask).resize((rend_size[1], rend_size[0]), Image.NEAREST)
    detected_lms[:, :] *= downsample_factor



    # adjust intrinsics accordingly
    if intrinsics_provided:
        K_hq = Intrinsics(K).crop(left_start, top_start)
        K = Intrinsics(K).crop(left_start, top_start).rescale(scale_factor=downsample_factor)
    else:
        K_hq = Intrinsics(K)
        K = Intrinsics(K).rescale(scale_factor=downsample_factor)

    # compute forground mask
    mouth_interior_mask = np.array(mouth_interior_mask)
    facer_mask = np.array(facer_mask)
    rgb_loss_mask = np.logical_not((facer_mask == 3) | (facer_mask <= 1))#| (facer_mask == max_class))
    foreground_mask = rgb_loss_mask | (facer_mask == 1)  | (facer_mask == 14)# include neck and hair
    facer_mask[~foreground_mask] = 0
    facer_mask_hq = np.array(facer_mask_hq)
    rgb_loss_mask = np.logical_not((facer_mask_hq == 3) | (facer_mask_hq <= 1))  # | (facer_mask == max_class))
    foreground_mask_hq = rgb_loss_mask | (facer_mask_hq == 1) | (facer_mask_hq == 14) # include neck and hair
    facer_mask_hq[~foreground_mask_hq] = 0

    # extract iBUG68 landmarks
    detected_lms = detected_lms[WFLW_2_iBUG68, :]

    data['landmarks_2d'].append(detected_lms)
    if intrinsics_provided:
        #fa_lms = np.load(
        #    f'{env_paths.DATA_TRACKING}/{seq_name}/kpt/{timestep:05d}.npy') * 2 #512 * 1080  # downsample_factor
        #fa_lms[:, 0] = fa_lms[:, 0] - left_start
        #fa_lms[:, 1] = fa_lms[:, 1] - top_start
        #fa_lms *= downsample_factor
#
        #im_tmp = np.array(rgb)
        #import cv2
#
        #for anchor_idx in range(detected_lms.shape[0]):
        #    im_tmp = cv2.circle(im_tmp, (
        #        int(detected_lms[anchor_idx][0].item()),
        #       int(detected_lms[anchor_idx][1].item())),
        #                       radius=1, color=(255, 0, 0), thickness=-1)
        #    im_tmp = cv2.circle(im_tmp, (
        #        int(fa_lms[anchor_idx][0].item()),
        #        int(fa_lms[anchor_idx][1].item())),
        #                        radius=5, color=(0, 255, 255), thickness=-1)
        #I = Image.fromarray(im_tmp)
        #I.show()
        #exit()

        #detected_lms[:, 0] -= left_start
        #detected_lms[:, 1] -= top_start
        fa_lms = None

    else:
        fa_lms = np.load(
            f'{env_paths.DATA_TRACKING}/{seq_name}/kpt/{timestep:05d}.npy')
        fa_lms[:, 0] = fa_lms[:, 0] - left_start
        fa_lms[:, 1] = fa_lms[:, 1] - top_start
        fa_lms *= downsample_factor



    #import cv2
    #im_tmp = np.array(rgb)
    #for anchor_idx in range(detected_lms.shape[0]):
    #    im_tmp = cv2.circle(im_tmp, (
    #        int(detected_lms[anchor_idx][0].item()),
    #        int(detected_lms[anchor_idx][1].item())),
    #                       radius=1, color=(255, 0, 0), thickness=-1)
    #    if fa_lms is not None:
    #        im_tmp = cv2.circle(im_tmp, (
    #            int(fa_lms[anchor_idx][0].item()),
    #            int(fa_lms[anchor_idx][1].item())),
    #                            radius=1, color=(0, 255, 255), thickness=-1)
    #I = Image.fromarray(im_tmp)
    #I.show()
    #im_tmp[foreground_mask] = 255
    #im_tmp[~foreground_mask] = 0
    #I = Image.fromarray(im_tmp)
    #I.show()

    if fa_lms is not None:
        detected_lms[:17, 0] = fa_lms[:17, 0]
        detected_lms[:17, 1] = fa_lms[:17, 1]


    K = np.array(K)
    K_hq = np.array(K_hq)


    # normalize rgb values into [-1, 1]
    rgb = ((np.array(rgb) / 255) - 0.5) * 2
    rgb_hq = ((np.array(rgb_hq) / 255) - 0.5) * 2


    # compute viewing dirctions for each ray in world space
    points3d_const_depth = back_project(np.ones_like(rgb[:, :, 0]) * 0.5,
                                        K,
                                        c2w_ex,
                                        rend_size=rend_size)
    view_dirs = points3d_const_depth - c2w_ex[:3, 3]
    view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    points3d_const_depth = back_project(np.ones_like(rgb_hq[:, :, 0]) * 0.5,
                                        K_hq,
                                        c2w_ex,
                                        rend_size=rend_size_hq)
    view_dirs_hq = points3d_const_depth - c2w_ex[:3, 3]
    view_dirs_hq /= np.linalg.norm(view_dirs_hq, axis=-1, keepdims=True)

    w, h = rend_size[1], rend_size[0]
    w_hq, h_hq = rend_size_hq[1], rend_size_hq[0]


    w2c_ex = c2w_ex.copy()
    w2c_ex[:3, :3] = np.linalg.inv(w2c_ex[:3, :3])
    w2c_ex[:3, 3] = w2c_ex[:3, :3] @ - w2c_ex[:3, 3]

    # linearize image and masks
    rgb = torch.from_numpy(rgb).reshape(-1, 3)
    mouth_interior_mask = torch.from_numpy(mouth_interior_mask).reshape(-1)
    facer_mask = torch.from_numpy(facer_mask).reshape(-1)
    facer_mask_hq = torch.from_numpy(facer_mask_hq).reshape(-1)


    view_dirs = torch.from_numpy(view_dirs)
    view_dirs_hq = torch.from_numpy(view_dirs_hq)
    cam_pos = torch.from_numpy(c2w_ex[:3, 3]) # camera position
    data['rgb'].append(rgb)
    data['segmentation_mask'].append(facer_mask)
    data['segmentation_mask_hq'].append(facer_mask_hq)
    data['view_dir'].append(view_dirs)
    data['view_dir_hq'].append(view_dirs_hq)
    data['cam_pos'].append(cam_pos)
    data['width'].append(w)
    data['width_hq'].append(w_hq)
    data['height'].append(h)
    data['height_hq'].append(h_hq)
    data['w2c'].append(w2c_ex)
    data['intrinsics'].append(K)
    data['intrinsics_hq'].append(K_hq)
    data['mouth_interior_mask'].append(mouth_interior_mask)

    return data
