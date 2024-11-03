import traceback
import pyvista as pv
from pyvista import global_theme

import trimesh
import cv2
import os
import json, csv
import point_cloud_utils as pcu
import time
import tyro
from mononphm import env_paths
from mononphm.utils_3d.coordinate_transforms import rigid_transform, similarity_transform, apply_transform, invert_similarity_transformation
from mononphm.utils_3d.render import render
from scipy.ndimage import binary_dilation
from dreifus.matrix import Intrinsics, Pose
from dreifus.camera import CameraCoordinateConvention
from mononphm.utils_3d.icp import get_ridid_alignment
from distinctipy import get_colors

import numpy as np
from PIL import Image



EVALUATIONS = f'{env_paths.TRACKING_OUTPUT}/../mononphm_evaluation/'
os.makedirs(EVALUATIONS, exist_ok=True)


def timeit(t0, tag):
    t = time.time()
    print(f'TIMETIMETIMEIT: {tag}: {t-t0}')
    return t


def start_xvfb(wait=3, window_size=None, display: int = 78, screen: int = 0):
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



EVAL_INTERVAL = 9

ANCHOR_iBUG68_pairs_39 = np.array([
    [0, 48], #left mouth corner
    [1, 64], #right mouth corner
    [35, 50], # upper middle lip
    [35, 52], # upper middle lip
    [34, 57], # lower middle lip
    [38, 8], # chin
    [33, 30], # nose
    [10, 39], # left eye middle corner
    [12, 36], # left eye outer corner
    [11, 42], # rigth eye middle corner
    [13, 45], # right corner outer corner
    [8, 19], # left eyebrow middle
    [9, 24], # right eye brow
    #[2, 2],
    #[3, 15],
    #[16, 4],
    #[16, 5],
    #[17, 12],
    #[17, 13],
])
ANCHOR_iBUG68_pairs_65 = np.array([
    # don't use jaw for rough rigid alignment
    #[0, 0], #left upmost jaw
    #[1, 16], #right upmost jaw
    #[38, 2], # jaw
    #[39, 14], # jaw
    #[2, 4], # jaw
    #[3, 12], # jaw
    #[4, 6], # jaw
    #[5, 10], # jaw
    #[60, 8], # chin

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

    # also don't use mouth
    #[16, 48], # l mouth corner
    #[17, 54], # r mouth corner
    #[18, 50], # l mouth top
    #[19, 52], # r mouth top
    #[20, 58], # l mouth bottom
    #[21, 56], # r mouth bottom,
    #[44, 49], # mouth
    #[45, 53],# mouth
    #[46, 59], # mouth
    #[47, 55],# mouth
    [48, 38],#eye
    [49, 43],#eye
    [50, 41],#eye
    [51, 46],#eye
    [52, 21],
    [53, 22]

])
WFLW_2_iBUG68 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]

lm_inds = np.array(
            [2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
                   3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
                   3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
                   3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
                   1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
                   1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
                   3509, 2786]
        )

def compute_normals(depth_gt, p3d):
    kernel_size = 11
    ddx2 = cv2.Sobel(p3d[:, :, 0], cv2.CV_64F, 1, 0, ksize=kernel_size)
    ddy2 = cv2.Sobel(p3d[:, :, 1], cv2.CV_64F, 0, 1, ksize=kernel_size)

    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    zx = cv2.Sobel(depth_gt, cv2.CV_64F, 1, 0, ksize=kernel_size)
    zy = cv2.Sobel(depth_gt, cv2.CV_64F, 0, 1, ksize=kernel_size)

    zy /= ddy2
    zx /= ddx2

    normal = np.dstack((-zx, -zy, np.ones_like(depth_gt)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal


def read_intrinsics_kinect_from_json(path_to_intrinsics_json, im_size=None, center_crop_fix_intrinsics=False,
                                     crop_details=None):
    # Kinect recording dumps the intrinsicscalibration to a json
    with open(path_to_intrinsics_json, 'r') as fp:
        print("Loading intrinsics from JSON")
        calibration = fp.readline().split()
        # Achtung! these correspond to the original image size it was captured with. cx,cy are in pixels.
        fx = float(calibration[2])
        fy = float(calibration[3])
        cx = float(calibration[0])
        cy = float(calibration[1])
        print("old cx cy: %d %d" % (cx, cy))

        if crop_details is not None:
            print("cropping images, adapting intrinsics")
            crop_start = crop_details['start']
            cx = calibration['cx'] - crop_start[1]
            cy = calibration['cy'] - crop_start[0]
            print("new cx cy: %d %d" % (cx, cy))
    print("Done.")

    return np.array([fx, fy, cx, cy])


def back_project_points(intrinsics,
                        img: np.ndarray,
                        depth_img: np.ndarray,
                        depth_cutoff: float = 1000,
                        return_colors: bool = False,
                        landmarks=None):  # -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    intrinsics_inv = np.linalg.inv(intrinsics)
    x_range = np.arange(img.shape[0])
    y_range = np.arange(img.shape[1])
    x_range = np.repeat(np.expand_dims(x_range, 1), img.shape[1], axis=1)
    y_range = np.repeat(np.expand_dims(y_range, 0), img.shape[0], axis=0)
    img_grid = np.stack([y_range, x_range], axis=-1)

    mask_valid_depths = depth_img < depth_cutoff

    if landmarks is None:
        depths = depth_img[mask_valid_depths]
        pixels = img_grid[mask_valid_depths]
        pixels = np.concatenate([pixels, np.ones_like(pixels[:, :1])], axis=-1)
        points_screen = pixels * np.expand_dims(depths, axis=-1)
        points_source_cam = points_screen @ intrinsics_inv.T
    else:
        depths = []
        for i in range(landmarks.shape[0]):
            depths.append(depth_img[landmarks[i, 1], landmarks[i, 0]])

        depths = np.stack(depths)
        pixels = landmarks
        pixels = np.concatenate([pixels, np.ones_like(pixels[:, :1])], axis=-1)
        points_screen = pixels * np.expand_dims(depths, axis=-1)
        points_source_cam = points_screen @ intrinsics_inv.T

    return points_source_cam



# computes metrics against Kinect point cloud
# saves "average_metrics.json" file for each sequence
def main(model_name : str,
         is_headless : bool = False,
         is_debug : bool = False,
         ):

    if is_headless:
        print('RUNNING ON SERVER; need to init virtual display')
        start_xvfb()


    # list testing sequences and the frame range which is used for evaluation
    # None means that the upper bound is the last frame of that sequence
    challeng_expressions = {
        '507_seq_1': (0, None),
        '507_seq_2': (23, None),
        '507_seq_3': (26, None),
        '507_seq_4': (0, None),

        '508_seq_1': (0, None),
        '508_seq_2': (24, None),
        '508_seq_3': (27, None),
        '508_seq_4': (25, None),

        '509_seq_1': (0, None),
        '509_seq_2': (0, None),
        '509_seq_3': (0, None),
        '509_seq_4': (0, None),

        '510_seq_1': (0, 170),
        '510_seq_2': (0, 180),
        '510_seq_3': (0, None),
        '510_seq_4': (0, None),

        '511_seq_1': (0, None),
        '511_seq_2': (0, None),
        '511_seq_3': (0, None),
        '511_seq_4': (0, None),
    }



    for seq_tag in challeng_expressions.keys():
        try:
            t0 = time.time()

            source = f'{env_paths.DATA_TRACKING}/{seq_tag}/'
            result_dir = f'{env_paths.TRACKING_OUTPUT}/{model_name}/'
            mica_tracker_dir = f'/{env_paths.DATA_TRACKING}/{seq_tag}/metrical_tracker/{seq_tag}/'

            intrinsics = read_intrinsics_kinect_from_json(f'{env_paths.ASSETS}/kinect_intrinsics.txt')
            K = np.eye(3)
            K[0, 0] = intrinsics[0]
            K[1, 1] = intrinsics[1]
            K[0, 2] = intrinsics[2]
            K[1, 2] = intrinsics[3]

            if is_headless:
                assert not is_debug

            filenames = os.listdir(os.path.join(source, 'depth'))
            filenames.sort(key=lambda x: int(x.split('.')[0]))
            files = os.listdir(f'{env_paths.DATA_TRACKING}/{seq_tag}/metrical_tracker/{seq_tag}/video/')
            n_expr = len(files)
            if challeng_expressions[seq_tag][1] is not None:
                n_expr = challeng_expressions[seq_tag][1]
            expressions = range(challeng_expressions[seq_tag][0], n_expr, 1)

            processed_timesteps = list(expressions)
            thresholds = [0.001, 0.0025, 0.005, 0.01]

            metrics_per_frame = {
                'chamfer_l1': [],
                'chamfer_l2': [],
                'normals': [],
                'fscore_0.001': [],
                'fscore_0.0025': [],
                'fscore_0.005': [],
                'fscore_0.01': [],
                'timesteps': []
            }

            t0 = timeit(t0, 'prepare stuff')

            for i, filename in enumerate(filenames):
                timestep = int(filename.split('.')[0])
                if timestep not in processed_timesteps:
                    continue
                if timestep % EVAL_INTERVAL != 0:
                    continue
                t0 = time.time()

                similarity_transforms  = {}

                print(filename)

                # load depth and convert to meters, and load color
                depth_gt = cv2.imread(os.path.join(source , 'depth' , filename), cv2.IMREAD_UNCHANGED)
                color_gt = cv2.imread(os.path.join(source , 'source' , filename))
                depth_gt = depth_gt / 1000 # to metres

                # load segmentation classes, needed to decide which regions are relevant for evaluation
                seg_classes = Image.open(f'{source}/seg/{timestep}.png')
                seg_classes = np.array(seg_classes.resize((1080, 1080), Image.NEAREST))
                neck_region = seg_classes == 1
                neck_region = binary_dilation(neck_region, np.ones([21, 21]))
                eye_region = (seg_classes == 8) | (seg_classes == 9)
                eye_region = binary_dilation(eye_region, np.ones([13, 13]))
                mouth_region = (seg_classes == 11)
                mouth_region = binary_dilation(mouth_region, np.ones([21, 21]))

                # compute mask used for evaluation
                face_mask = (seg_classes == 2) | (seg_classes==6) | (seg_classes==7) | (seg_classes==10) | (seg_classes==12) | (seg_classes==13)
                face_mask = face_mask & (~neck_region) & (~mouth_region) & (~eye_region)
                face_mask = face_mask.reshape(-1)
                lms2d = np.load(f'{source}/pipnet/test.npy')[timestep, :]
                lms2d[:, 0] *= depth_gt.shape[1]
                lms2d[:, 1] *= depth_gt.shape[0]

                lms2d = lms2d[WFLW_2_iBUG68, :]

                # depth to point cloud
                points3d_dreifus = back_project_points(K, color_gt, depth_gt)
                points3d_constdepth = back_project_points(K, color_gt, np.ones_like(depth_gt))
                landmarks3d_dreifus = back_project_points(K, color_gt, depth_gt, landmarks=lms2d.astype(int))

                # get normals, then filter points with a steep obersvation angle (angle betwwn normal and viewing dir)
                normals = compute_normals(depth_gt, points3d_dreifus.reshape([depth_gt.shape[0], depth_gt.shape[1], 3])).reshape(-1, 3)
                view_dir = points3d_constdepth - 0
                view_dir = view_dir / np.linalg.norm(view_dir, axis=-1, keepdims=True)
                view_surf_angle = np.sum(view_dir * normals, axis=-1)


                normals_rgb = ((normals + 1)/2*255).astype(np.uint8)


                forground = (depth_gt.reshape(-1) < 0.8) & (depth_gt.reshape(-1) > 0.01)
                valid_landmarks_dreifus = (landmarks3d_dreifus[:, 2] < 0.8) & (landmarks3d_dreifus[:, 2] > 0.01)


                valid = forground & (view_surf_angle > 0.2)


                # Visualize back-project kinect point cloud before (left) and after (right) normal based filtering
                if is_debug:
                    pl = pv.Plotter(shape=(1, 2))
                    pl.subplot(0, 0)
                    pl.add_points(points3d_dreifus[forground], scalars=view_surf_angle[forground])
                    pl.subplot(0, 1)
                    pl.add_points(points3d_dreifus[valid], scalars=view_surf_angle[valid])
                    pl.link_views()
                    pl.show()

                forground = valid

                dist_colors = get_colors(25)
                seg_classes = seg_classes.reshape(-1)
                seg_classes = seg_classes[forground]
                colors = np.zeros_like(points3d_dreifus[forground]).astype(np.uint8)
                for i in range(25):
                    colors[seg_classes==i] = (np.array(dist_colors[i])*255).astype(np.uint8)


                # visualize semantic classes on top of point cloud (left) and visualize face region (right),
                #   which is used for evaluation. Due to noisy measurements inside of the eyes and the mouth,
                #   these regions are removed from the point cloud
                if is_debug:
                    pl = pv.Plotter(shape=(1, 2))
                    pl.subplot(0, 0)
                    pl.add_points(points3d_dreifus[forground], scalars=colors, rgb=True)
                    pl.subplot(0, 1)
                    pl.add_points(points3d_dreifus[forground&face_mask], scalars=normals_rgb[forground&face_mask], rgb=True)
                    pl.link_views()
                    pl.show()






                t0 = timeit(t0, 'kinect loading')

                # load tracked FLAME mesh and its head pose, move it to camera space
                # note however that due to scale ambiguity, this alignment is not well-determined
                # next we will use FLAME landmarks and 3d landmarks to improve the alignment
                mica_pose = np.load(
                    f'{env_paths.DATA_TRACKING}/{seq_tag}/metrical_tracker/{seq_tag}//checkpoint/{max(timestep - 1, 0):05d}_cam_params_opencv.npz')


                mesh_mica_tracker = trimesh.load(f'{mica_tracker_dir}/mesh/{timestep:05d}.ply', process=False, maintain_order=True)
                mesh_mica_tracker.vertices = mesh_mica_tracker.vertices @ mica_pose['R'][0].T + mica_pose['t'][0]
                sim_mica = np.eye(4)
                sim_mica[:3, :3] = mica_pose['R'][0]
                sim_mica[:3, 3] = mica_pose['t'][0]
                similarity_transforms['mica_tracker'] = sim_mica

                mesh_flame = mesh_mica_tracker


                c2w_pose = Pose(mica_pose['R'][0], mica_pose['t'][0],
                                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                c2w_pose = c2w_pose.invert().change_camera_coordinate_convention(
                    new_camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL).invert()
                rot = np.array(c2w_pose)[:3, :3]
                trans = np.array(c2w_pose)[:3, 3]


                # load tracked NPHM mesh, and it's pose, relative to the FLAME pose
                mesh_nphm = trimesh.load(f'{result_dir}/stage2/{seq_tag}/{timestep:05d}/mesh.ply', process=False)
                s = np.squeeze(np.load(f'{result_dir}/stage2/{seq_tag}/{timestep:05d}//scale.npy'))
                R = np.squeeze(np.load(f'{result_dir}/stage2/{seq_tag}/{timestep:05d}//rot.npy'))
                t = np.squeeze(np.load(f'{result_dir}/stage2/{seq_tag}/{timestep:05d}//trans.npy'))

                anchors = None

                sim_nphm1 = np.eye(4)
                sim_nphm1[:3, :3] = 4*s*R
                sim_nphm1[:3, 3] = t

                # transform nphm mesh to FLAME coordinates and FLAME head pose
                mesh_nphm.vertices = 1/s *  (mesh_nphm.vertices - t) @ R
                mesh_nphm.vertices /= 4

                # transform from FLAME coordinates to camera space
                mesh_nphm.vertices = mesh_nphm.vertices @ rot.T + trans

                sim_nphm2 = np.eye(4)
                sim_nphm2[:3, :3] = rot
                sim_nphm2[:3, 3] = trans

                # obtain 4x4 matrix that describes the above two-step transformation in one
                sim_combo = sim_nphm2 @ invert_similarity_transformation(sim_nphm1)
                similarity_transforms = sim_combo

                # also transform anchors in the same way
                if anchors is not None:
                    anchors = 1/s *  (anchors-t) @ R
                    anchors /= 4
                    anchors = anchors @ rot.T + trans


                # OpenVC to OpenGL
                mesh_nphm.vertices[:, 2] *= -1
                mesh_nphm.vertices[:, 1] *= -1
                sim = np.eye(4)
                sim[:, 2] *= -1
                sim[:, 1] *= - 1
                similarity_transforms = sim @ similarity_transforms
                if anchors is not None:
                    anchors[:, 2] *= -1
                    anchors[:, 1] *= -1



                # Visualize point cloud, mesh from metrical tracker and the camera center
                #if DEBUG:
                #    pl = pv.Plotter()
                #    pl.add_points(points3d_dreifus[forground])
                #    pl.add_mesh(mesh_flame)
                #    #if mesh_nphm is not None:
                #    #    pl.add_mesh(mesh_nphm, color='green')
                #    pl.add_points(np.zeros([3]), color='red')
                #    pl.show()

                landmarks_flame = mesh_flame.vertices[lm_inds, :]

                # visualize the landmarks of FLAME and backprojected detected landmarks.
                #   These are used to compute a coarse similarity transform in order to account for depth ambiguity
                #if DEBUG:
                #    pl = pv.Plotter()
                #    valid_lm_inds = valid_landmarks_dreifus[17:48]
                #    pl.add_points(landmarks_flame[17:48, :][valid_lm_inds, :])
                #    pl.add_points(landmarks3d_dreifus[17:48, :][valid_lm_inds, :], color='red')
                #    for i in range(17, 48):
                #        if valid_landmarks_dreifus[i]:
                #            pl.add_mesh(pv.Line(landmarks_flame[i], landmarks3d_dreifus[i]))
                #    pl.show()

                # compute coarse alignment based on landmarks to fix scale ambiguity
                # the alignment is realiszed using a similarity transform using the method from Umeyama
                # However, it seems that this coarse alignment is not needed
                valid_lm_inds = valid_landmarks_dreifus[17:48]
                _landmarks_flame = landmarks_flame[17:48, :][valid_lm_inds, :]
                _landmarks3e_dreifus = landmarks3d_dreifus[17:48, :][valid_lm_inds, :]
                lm_dist = np.linalg.norm(_landmarks_flame - _landmarks3e_dreifus, axis=-1)
                median_dist = np.median(lm_dist)
                invalid_matches = lm_dist > median_dist * 1.2
                _landmarks_flame = _landmarks_flame[~invalid_matches]
                _landmarks3e_dreifus = _landmarks3e_dreifus[~invalid_matches]
                _, R, t = rigid_transform(_landmarks_flame.T, _landmarks3e_dreifus.T)

                landmarks_flame = landmarks_flame @ R.T + t


                #if DEBUG:
                #    pl = pv.Plotter()
                #    valid_lm_inds = valid_landmarks_dreifus[17:]

                    #pl.add_points(landmarks_flame[17:, :][valid_lm_inds, :])
                    #pl.add_points(landmarks3d_dreifus[17:, :][valid_lm_inds, :], color='red')
                    #pl.show()



                correctiveT = np.eye(4)
                correctiveT[:3, :3] = R
                correctiveT[:3, 3] = t

                mesh_flame = apply_transform(mesh_flame.copy(), correctiveT)

                # viz results after coarse alignment
                #if DEBUG:
                #    pl = pv.Plotter()
                #    pl.add_mesh(mesh_flame)
                #    pl.add_mesh(mesh_mica_tracker, color='red')
                #    pl.add_points(points3d_dreifus[forground & face_mask])
                #    pl.show()


                t0 = timeit(t0, 'loading meshes and rough alignment')

                t0 = time.time()


                opengl_iden_cam = Pose(np.eye(4), camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)


                # Render reconstructed mesh
                pl = None
                image = render(pl, mesh_nphm, opengl_iden_cam, Intrinsics(K), wh=[1080, 1080], light_pose_zero=True, )
                I = Image.fromarray(image)
                os.makedirs(f'{EVALUATIONS}/{model_name}/{seq_tag}_images/', exist_ok=True)
                I.save(f'{EVALUATIONS}/{model_name}/{seq_tag}_images/{timestep:05d}.png')


                t0 = timeit(t0, 'render image')
                print('hi')


                # only eval every n-th step for computational reasons
                if timestep % EVAL_INTERVAL == 0:
                    # compute a similarity transform using ICP
                    rigid_T_icp = get_ridid_alignment(mesh_nphm, points3d_dreifus[forground & face_mask],
                                                      -1 * normals[forground & face_mask], is_flame = False)

                    rigid_T_icp_4x4 = np.eye(4)
                    rigid_T_icp_4x4[:3, :3] = rigid_T_icp['s']*rigid_T_icp['R']
                    rigid_T_icp_4x4[:3, 3] = rigid_T_icp['t']
                    similarity_transforms = rigid_T_icp_4x4 @ similarity_transforms
                    mesh_nphm.vertices = rigid_T_icp['s'] * mesh_nphm.vertices @ rigid_T_icp['R'].T + rigid_T_icp['t']

                    # viz results after ICP-style alignment, we show the reconstructed mesh,
                    #   the kinect point cloud and camera center
                    if is_debug:
                        pl = pv.Plotter()
                        pl.add_points(points3d_dreifus[forground & face_mask])
                        pl.add_mesh(mesh_nphm)
                        pl.add_points(np.zeros([3]), color='red')
                        pl.show()

                    # compute the chamfer distance from point cloud to reconstructed mesh
                    d, fi, bc = pcu.closest_points_on_mesh(points3d_dreifus[forground & face_mask], mesh_nphm.vertices, mesh_nphm.faces)

                    # Convert barycentric coordinates to 3D positions
                    closest_points = pcu.interpolate_barycentric_coords(mesh_nphm.faces, fi, bc, mesh_nphm.vertices)

                    dist_chamferl1 = np.abs(points3d_dreifus[forground & face_mask] - closest_points).sum(axis=-1)
                    dist_chamferl2 = np.linalg.norm(points3d_dreifus[forground & face_mask] - closest_points,
                                                axis=-1)
                    rec_paired_normals= mesh_nphm.face_normals[fi]



                    f_scores = []
                    for thresh in thresholds:
                        f_scores.append(dist_chamferl2 <= thresh)

                    normals_pred = rec_paired_normals / np.linalg.norm(rec_paired_normals, axis=-1, keepdims=True)
                    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

                    normals_dot_product = (normals[forground & face_mask] * normals_pred).sum(axis=-1)
                    normals_dot_product = np.abs(normals_dot_product)

                    # visualize the kinect point cloud, as well as, point-cloud-to-mesh chamfer distance
                    if is_debug:
                        pl = pv.Plotter()
                        pl.add_points(points3d_dreifus[forground & face_mask], scalars=normals_dot_product) #dist_chamferl1)
                        pl.show()

                    os.makedirs(f'{EVALUATIONS}/{model_name}/{seq_tag}_evaluation/', exist_ok=True)
                    np.savez(f'{EVALUATIONS}/{model_name}/{seq_tag}_evaluation/error_pc_{timestep:05d}.npz',
                             points=points3d_dreifus[forground & face_mask],
                             chamfer_l1=dist_chamferl1,
                             chamfer_l2=dist_chamferl2,
                             normals=normals_dot_product)

                    os.makedirs(f'{EVALUATIONS}/{model_name}/{seq_tag}_transforms/', exist_ok=True)
                    np.save(f'{EVALUATIONS}/{model_name}/{seq_tag}_transforms/{timestep:05d}.npy', similarity_transforms)


                    # render overlay of reconstructed mesh and point cloud with chamfer-loss colors
                    pl = None
                    image = render(pl, mesh_nphm, opengl_iden_cam, Intrinsics(K), wh=[1080, 1080],
                                   light_pose_zero=True,
                                   pointcloud=points3d_dreifus[forground & face_mask], pointcloud_scalars=dist_chamferl2)
                    I = Image.fromarray(image)
                    # I.show()
                    os.makedirs(f'{EVALUATIONS}/{model_name}/{seq_tag}_images_errorOverlay/', exist_ok=True)
                    I.save(f'{EVALUATIONS}/{model_name}/{seq_tag}_images_errorOverlay/{timestep:05d}.png')

                    # render only the point cloud with chamfer-loss colors
                    pl = None
                    image = render(pl, None, opengl_iden_cam, Intrinsics(K), wh=[1080, 1080],
                                   light_pose_zero=True,
                                   pointcloud=points3d_dreifus[forground & face_mask],
                                   pointcloud_scalars=dist_chamferl2)
                    I = Image.fromarray(image)
                    os.makedirs(f'{EVALUATIONS}/{model_name}/{seq_tag}_images_error/', exist_ok=True)
                    I.save(f'{EVALUATIONS}/{model_name}/{seq_tag}_images_error/{timestep:05d}.png')


                    dist_chamferl1 = dist_chamferl1.mean()
                    dist_chamferl2 = dist_chamferl2.mean()
                    normals_dot_product = normals_dot_product.mean()
                    f_scores = [f.mean() for f in f_scores]
                    metrics = {
                        'chamfer_l1': f'{dist_chamferl1:2.6f}',
                        'chamfer_l2': f'{dist_chamferl2:2.6f}',
                        'normals': f'{normals_dot_product:2.6f}',

                    }

                    for ti, thresh in enumerate(thresholds):
                        metrics[f'fscore_{thresh}'] = f'{f_scores[ti]:2.6f}'

                    metrics_per_frame['timesteps'].append(timestep)
                    metrics_per_frame['chamfer_l1'].append(dist_chamferl1)
                    metrics_per_frame['chamfer_l2'].append(dist_chamferl2)
                    metrics_per_frame['normals'].append(normals_dot_product)
                    for ti, thresh in enumerate(thresholds):
                        metrics_per_frame[f'fscore_{thresh}'].append(f_scores[ti])

                    t0 = timeit(t0, 'metrics')


            metrics_per_frame = {k: np.array(metrics_per_frame[k]) for k in metrics_per_frame.keys()}


            np.savez(f'{EVALUATIONS}/{model_name}/{seq_tag}_evaluation/all_metrics.npz',
                     chamfer_l1=metrics_per_frame['chamfer_l1'],
                     chamfer_l2=metrics_per_frame['chamfer_l2'],
                     normals=metrics_per_frame['normals'],
                     fscore_001=metrics_per_frame['fscore_0.001'],
                     fscore_0025=metrics_per_frame['fscore_0.0025'],
                     fscore_005=metrics_per_frame['fscore_0.005'],
                     fscore_01=metrics_per_frame['fscore_0.01'],
                     )
            avg_metrics = {k: f'{np.nanmean(metrics_per_frame[k]):2.6f}' for k in metrics_per_frame.keys()}

            with open(f'{EVALUATIONS}/{model_name}/{seq_tag}_evaluation/avg_metrics.json', 'w') as fp:
                json.dump(avg_metrics, fp)

            with open(f'{EVALUATIONS}/{model_name}/{seq_tag}_evaluation/avg_metrics.csv', "w") as file:
                writer = csv.writer(file)
                writer.writerow([avg_metrics[k] for k in avg_metrics.keys()])
        except Exception as ex:

            traceback.print_exc()


if __name__ == '__main__':
    tyro.cli(main)