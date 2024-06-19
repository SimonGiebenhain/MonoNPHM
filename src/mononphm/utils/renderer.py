import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import pyrender
import trimesh


def m3dLookAt(eye, target, up):
    mz = (eye - target)
    mz /= np.linalg.norm(mz, keepdims=True)  # inverse line of sight
    mx = np.cross(up, mz)
    mx /= np.linalg.norm(mx, keepdims=True)
    my = np.cross(mz, mx)
    my /= np.linalg.norm(my)
    tx = eye[0]  # np.dot(mx, eye)
    ty = eye[1]  # np.dot(my, eye)
    tz = eye[2]  # -np.dot(mz, eye)
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
    #ppos[:, 0] = np.clip(ppos[:, 0], 0, 3840)  # 1279) #
    #ppos[:, 1] = np.clip(ppos[:, 1], 0, 2160)  # 959) #
    points_depth = depth[ppos[:, 1], ppos[:, 0]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN

    return points, points_depth


def render(mesh, K, cam_pose, rend_size, render_flat=True):
    # sc = trimesh.Scene([mesh])
    # sc.show(flags={'flat': True})
    # mesh.show()
    try:
        mat = mesh.visual.material
        glossiness = mat.kwargs.get('Ns', 1.0)
        mat_img = mat.image
    except Exception as e:
        glossiness = 1.0
        mat_img = None
    if isinstance(glossiness, list):
        glossiness = float(glossiness[0])
    roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)

    material = pyrender.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        roughnessFactor=roughness,
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=mat_img,
    )

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material)
    scene = pyrender.Scene()  # ambient_light=[0., 0., 0., .0])  # 0.15 for .ply

    # Adding objects to the scene
    scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy =  K[1][2]

    znear = 0.1 #0.1  # 0.1 # 200 #
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

    #import pyvista as pv
    #pl = pv.Plotter()
    #pl.add_mesh(mesh)
    #pl.add_points(cam_pose[:3, 3])
    #pl.show()

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0])
    if render_flat:
        color, depth = r.render(scene, pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)

    depth[depth == 0] = float('inf')
    depth_og = depth.copy()
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)

    return depth, color, depth_og


def back_project(depth, K, cam_pose, rend_size=(512, 512)):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
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
    points, _ = unproject_points(lm_inds[:, :2], depth, rend_size,
                                 cam.get_projection_matrix(rend_size[1], rend_size[0]), cam_pose)

    return points


def render_and_backproject(subject, expression,
                           path_mesh,
                           path_registration,
                           slice_mesh=True,
                           crop=50,
                           res_factor = 2,
                           eye = 4 * np.array([0, 0, 0.65])):
    #rend_size = (1280 // res_factor, 960 // res_factor)
    rend_size = (960 // res_factor, 960 // res_factor)
    print('RENDERING SIZE:', rend_size)
    crop = crop // res_factor

    KK = np.array(
        [[2440 / res_factor, 0.00000000e+00, (rend_size[1] / 2) - crop],
         [0.00000000e+00, 2440 / res_factor, (rend_size[0] / 2) - crop],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )
    rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)

    mesh2 = trimesh.load(path_mesh)

    if path_registration is not None:
        mesh = trimesh.load(
            path_registration + '/warped.ply',
            process=False)
        s = np.load(path_registration + '/s.npy')
        t = np.load(path_registration + '/t.npy')
        R = np.load(path_registration + '/R.npy')

        mesh2.vertices = 4 * (s * mesh2.vertices @ R.T + t)
        mesh.vertices *= 4 #/= 25

    manager = DataManager()
    valid = manager.cut_throat(mesh2.vertices, subject=subject, expression=expression, coordinate_system='nphm', margin=0.001)
    invalid = np.logical_not(valid)
    invalid_faces = mesh2.vertex_faces[invalid, :]
    invalid_faces = np.unique(invalid_faces.reshape([-1]))
    invalid_faces = invalid_faces[invalid_faces >= 0]
    invalid_faces_mask = np.zeros(dtype=bool, shape=[mesh2.faces.shape[0]])
    invalid_faces_mask[invalid_faces] = 1
    mesh2.update_faces(np.logical_not(invalid_faces_mask))
    mesh2.remove_unreferenced_vertices()



    # TODO
    E = m3dLookAt(eye,
                  np.array([0, 0, 0]),  # np.mean(mesh2.vertices, axis=0),
                  np.array([0, 1, 0]))

    # convert from c2w to w2c camera pose
    EX_tmp = E.copy()
    # EX_tmp[:3, :3] = np.linalg.inv(EX_tmp[:3, :3])
    # EX_tmp[:3, 3] = EX_tmp[:3, :3] @ (-EX_tmp[:3, 3])

    # rendering
    depth_img, rend_img, depth_og = render(mesh2, KK, EX_tmp, rend_size=rend_size)
    I_rend = Image.fromarray(rend_img)
    # g.t. rendering
    np.save(f'/home/giebenhain/photometric_optimization/K_s{subject}_e{expression}.npy', KK)
    np.save(f'/home/giebenhain/photometric_optimization/E_s{subject}_e{expression}.npy', EX_tmp)
    I_rend.save(f'/home/giebenhain/photometric_optimization/FFHQ/s{subject}_e{expression}.png')
    print('SAVED CAM PARAMS FOR FLAME FITTING')
    #I_rend.show()

    # backproject
    points3d = back_project(depth_img, KK, EX_tmp, rend_size=rend_size)
    points3d_const_depth = back_project(np.ones_like(depth_img)*np.nanmean(depth_img), KK, EX_tmp, rend_size=rend_size)
    view_dirs = points3d_const_depth - E[:3, 3]
    view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)


    # save neural "rendering"
    return I_rend, points3d, view_dirs, mesh2, rend_size, depth_og, EX_tmp, KK

