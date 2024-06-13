import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import pyrender
import trimesh

from mononphm import env_paths

def m3dLookAt(eye, target, up):
    mz = (eye-target)
    mz /= np.linalg.norm(mz, keepdims=True)  # inverse line of sight
    mx = np.cross(up, mz)
    mx /= np.linalg.norm(mx, keepdims=True)
    my = np.cross(mz, mx)
    my /= np.linalg.norm(my)
    tx = eye[0] #np.dot(mx, eye)
    ty = eye[1] #np.dot(my, eye)
    tz = eye[2] #-np.dot(mz, eye)
    return np.array([[mx[0], my[0], mz[0], tx],
                     [mx[1], my[1], mz[1], ty],
                     [mx[2], my[2], mz[2], tz],
                     [0, 0, 0, 1]])

def unproject_points(ppos, depth, rend_size, K, Rt):
    points = np.ones((ppos.shape[0], 4))
    points[:, [0, 1]] = ppos.astype(float)
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    ppos[:, 0] = np.clip(ppos[:, 0], 0, 3840) # 1279) #
    ppos[:, 1] = np.clip(ppos[:, 1], 0, 2160) # 959) #
    points_depth = depth[ppos[:, 1], ppos[:, 0]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN
    #pl = pv.Plotter()
    #pl.add_points(points)
    #pl.show()

    return points, points_depth



def render(mesh, K, cam_pose, rend_size):

    mat = mesh.visual.material
    glossiness = mat.kwargs.get('Ns', 1.0)
    if isinstance(glossiness, list):
        glossiness = float(glossiness[0])
    roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)

    material = pyrender.MetallicRoughnessMaterial(
                        alphaMode='BLEND',
                        roughnessFactor=roughness,
                        baseColorFactor=[255, 255, 255, 255],
                        baseColorTexture=mat.image,
                    )

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material)
    scene = pyrender.Scene()#ambient_light=[0., 0., 0., .0])  # 0.15 for .ply

    # Adding objects to the scene
    scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = (rend_size[1] - K[0][2])
    cy = K[1][2]

    znear = 0.1  # 0.1 # 200 #
    zfar = 10.0

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)

    scene.add(cam, pose=cam_pose)

    # Set up the light
    instensity = 0.5
    light1 = pyrender.PointLight(intensity=instensity)
    light2 = pyrender.PointLight(intensity=instensity)

    light_pose1 = m3dLookAt(cam_pose[:3, 3] / 2 + np.array([0, 0, 300]), np.mean(mesh.vertices, axis=0),
                            up=np.array([0, 1, 0]))
    light_pose2 = m3dLookAt(cam_pose[:3, 3] / 2 + np.array([0, 0, 0]), np.mean(mesh.vertices, axis=0),
                            up=np.array([0, 1, 0]))

    light_pose1[:3, 3] = cam_pose[:3, 3] / 2 + np.array([0.15, 0.1, -0.15])
    light_pose2[:3, 3] = cam_pose[:3, 3] / 2 + np.array([-0.15, 0.1, -0.15])

    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0])
    color, depth = r.render(scene, pyrender.constants.RenderFlags.FLAT)

    depth[depth == 0] = float('inf')
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)


    return depth, color


def back_project_(depth, K, cam_pose, rend_size=(512, 512)):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = (960 - K[0][2])
    cy = K[1][2]

    znear = 0.1  # 0.1 # 200 #
    zfar = 10.0

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)

    xx, yy = np.meshgrid(np.arange(rend_size[0]), np.arange(rend_size[1]))
    points_positions = np.stack([yy, xx]).astype(np.int32)
    points = np.reshape(points_positions, [-1, 2])
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    points_depth = depth[xx.reshape([-1]), yy.reshape([-1])][:, np.newaxis]
    points = np.concatenate([points, points_depth, np.ones_like(points_depth)], axis=1)
    #points = np.concatenate([points, points_depth], axis=1)
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(cam_pose, np.linalg.inv(cam.get_projection_matrix(rend_size[1], rend_size[0])))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN

    points = points[~np.isnan(points[:, 0]), :]
    pl = pv.Plotter()
    pl.add_points(points)
    pl.show()

    return points


def back_project(depth, K, cam_pose, rend_size=(512, 512)):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = (rend_size[1] - K[0][2])
    cy = K[1][2]

    znear = 0.1  # 0.1 # 200 #
    zfar = 10.0

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)
    xx, yy = np.meshgrid(np.arange(rend_size[1]), np.arange(rend_size[0]))
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    lm_inds = np.stack([xx, yy], axis=-1).astype(np.int32)
    points, _ = unproject_points(lm_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), cam_pose)

    return points


def render_and_backproject(path_mesh, path_registration, res_factor):

    rend_size = (1280 // res_factor, 960 // res_factor)

    KK = np.array(
        [[2.44063982e+03 / res_factor, 0.00000000e+00, rend_size[1] / 2],
         [0.00000000e+00, 2.44144332e+03 / res_factor, rend_size[0] / 2],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )

    mesh = trimesh.load(
        path_registration + '/warped.ply',
        process=False)
    mesh2 = trimesh.load(path_mesh)
    s = np.load(path_registration + '/s.npy')
    t = np.load(path_registration + '/t.npy')
    R = np.load(path_registration + '/R.npy')

    mesh2.vertices = 4 * (s * mesh2.vertices @ R.T + t)
    mesh.vertices /= 25

    lm_inds_up = np.load(env_paths.flame_lm_indices_upsampled)
    E = m3dLookAt(4 * np.array([-0.00818554, -0.0196824, 0.657607]),
                  mesh.vertices[lm_inds_up[30], :],  # np.mean(mesh2.vertices, axis=0),
                  np.array([-0.00713758, 0.999157, 0.0404241]))

    # convert from c2w to w2c camera pose
    EX_tmp = E.copy()
    #EX_tmp[:3, :3] = np.linalg.inv(EX_tmp[:3, :3])
    #EX_tmp[:3, 3] = EX_tmp[:3, :3] @ (-EX_tmp[:3, 3])

    # rendering
    depth_img, rend_img = render(mesh2, KK, EX_tmp, rend_size=rend_size)
    # g.t. rendering
    I_rend = Image.fromarray(rend_img)

    # backproject
    points3d = back_project(depth_img, KK, EX_tmp,rend_size=rend_size)

    # save neural "rendering"
    return I_rend, points3d, mesh2, rend_size


if __name__ == '__main__':
    import pyvista as pv


    res_factor = 2


    path_mesh = '/mnt/hdd/scans/scans_5/id165_merged/fusion_watertight_textured000_down_tex.ply'
    path_reg = '/mnt/rohan/rhome/sgiebenhain/non-rigid-registration/scanner/FLAME/id165_merged/expression_0/'
    I_rend, _points3d, mesh2, rend_size = render_and_backproject(path_mesh, path_reg, res_factor=res_factor)

    I_rend.show()
    points3d = _points3d.copy()
    print(points3d.shape)
    points3d = points3d.reshape([rend_size[0], rend_size[1], 3])
    points3d[np.isnan(points3d[:, :, 0]), :] = np.nanmin(points3d)
    def normalize(arr):
        arr = (arr - np.min(arr, axis=(0, 1), keepdims=True))
        return arr / np.max(arr, axis=(0, 1), keepdims=True)

    img = np.ones([rend_size[0], rend_size[1], 3]) * 255
    img[:, :, :] = 255 * normalize(points3d)
    img = img.astype(np.uint8)
    I = Image.fromarray(img)
    I.show()



    pl = pv.Plotter()
    pl.add_points(_points3d)
    pl.add_mesh(mesh2)
    pl.show()


