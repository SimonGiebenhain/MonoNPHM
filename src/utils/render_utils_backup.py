import numpy as np
import pyvista as pv
import os
import pathlib
from scipy.spatial.transform import Rotation
from PIL import Image
import trimesh
import pyrender

from dreifus.matrix import Pose, Intrinsics
from dreifus.camera import PoseType, CameraCoordinateConvention
import math
import distinctipy
import cv2

orbit = 'upper'
scan_id = 0
KK = np.array([
    [2440, 0, 480],
    [0, 2440, 640],
    [0, 0, 1]
], dtype=np.float32)

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.vert",
                                                                 str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.frag",
                                                                 defines=defines)
        return self.program
    def clear(self):
        self.program = None

def render_glcam(model_in,  # model name or trimesh
                 K,
                 Rt,
                 lms,
                 rend_size=(512, 512),
                 show=False,
                 mm_scale=False):
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    #print('mesh copy: {}'.format(time()-t0))

    #material = pyrender.material.MetallicRoughnessMaterial(alphaMode='BLEND', baseColorFactor=[0.3, 0.3, 0.3, 1.0],
    #                                                       metallicFactor=0.00001, roughnessFactor=0.999999)
    material = pyrender.material.Material(alphaMode='BLEND')
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)#, material=material)
    #print('mesh to pyrender: {}'.format(time()-t0))
    # Scene creation
    #scene = pyrender.Scene(ambient_light = [0.7,0.7,0.7, 1.0])
    scene = pyrender.Scene(ambient_light = [0.45,0.45,0.45, 1.0]) # 0.15 for .ply

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    #cx, cy = K[0][2], K[1][2]
    #cx = K[0][2] #cx = (960 - K[0][2])
    cx = (960 - K[0][2])
    cy = K[1][2]
####
    znear = 0.1#0.1 # 200 #
    zfar = 2.0
    if mm_scale:
        znear = 200
        zfar = 2000#20 # 2000 #
    #znear = 0.001
    #zfar = 1000
    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)
    #pose = Pose(Rt[:3, :3], Rt[:3, 3])

    cam_pose = np.eye(4)
    Rt = np.array(Rt)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T @ Rt[:, 3]
    #cam_pose = pose.invert()
    scene.add(cam, pose=cam_pose)

    # Set up the light
    instensity = 0.75
    if mm_scale:
        instensity = 2000000
    light1 = pyrender.PointLight(intensity=instensity)
    light2 = pyrender.PointLight(intensity=instensity)

    light_pose1 = m3dLookAt(cam_pose[:3, 3]/2 + np.array([0, 0, 300]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))
    light_pose2 = m3dLookAt(cam_pose[:3, 3]/2 + np.array([0, 0, 0]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))

    light_pose1[:3, 3] = cam_pose[:3, 3]/2 + np.array([0.15, 0.1, -0.15])
    light_pose2[:3, 3] = cam_pose[:3, 3]/2 + np.array([-0.15, 0.1, -0.15])
    if mm_scale:
        light_pose1[:3, 3] = cam_pose[:3, 3] / 2
        light_pose2[:3, 3] = cam_pose[:3, 3] / 2

    #pl = pv.Plotter()
    #pl.add_mesh(mesh)
    #pl.add_points(light_pose1[:3, 3])
    #pl.add_points(light_pose2[:3, 3])
    #pl.add_points(cam_pose[:3, 3], color='red')
    #pl.add_mesh(pv.Line(light_pose1[:3, 3], light_pose1[:3, 3] + np.array([0, 200, 0])), color='red')
    #pl.add_mesh(pv.Line(light_pose1[:3, 3], light_pose1[:3, 3] + np.array([200, 0, 0])), color='blue')
    #pl.add_mesh(pv.Line(light_pose1[:3, 3], light_pose1[:3, 3] + np.array([0, 0, 200])), color='green')
    #pl.show()

    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)


    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)

    #r._renderer._program_cache = ShaderProgramCache(shader_dir="shaders")
    r._renderer._program_cache = CustomShaderCache()


    #color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)


    normals, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    r.delete()
    world_space_normals = normals / 255 * 2 - 1
    color=normals

    #image = Image.fromarray(normals, 'RGB')
    #image.show()


    depth[depth == 0] = float('inf')
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)

    #points = np.ones((indices.shape[0], 4))
    #points[:, [1, 0]] = indices.astype(float)
    #points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    #points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1
    #points[:, 1] *= -1
    #points[:, 2] = depth[indices[:, 0], indices[:, 1]]
    #K = cam.get_projection_matrix(rend_size[1], rend_size[0])
    #clipping_to_world = np.matmul(cam_pose, np.linalg.inv(K)) # TODO !!!
    #points = np.matmul(points, clipping_to_world.transpose())
    #points /= points[:, 3][:, np.newaxis]
    #points = points[:, :3]
    #points = unproject_points(indices, depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), cam_pose)
    #pl.add_points(points)

    #if lms is not None:
    #    lm_ppos = (lms[:, :2] - 0.5)*2
    #    lm_inds = lms.copy()
    #    lm_inds[:, 0] = lms[:, 1] * rend_size[0]
    #    lm_inds[:, 1] = lms[:, 0] * rend_size[1]
    #    lm_inds = np.floor(lm_inds).astype(np.int32)
    #    #lm_depth = depth[lm_inds[:, 0], lm_inds[:, 1]]
    #    lms3d, lms_depth = unproject_points(lm_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), cam_pose)
#
    #    if show:
    #        pl.add_points(lms3d, color='red', point_size=7)
#
    #        depth = depth / np.max(depth)
    #        for kk in range(lm_inds.shape[0]):
    #            depth = cv2.circle(depth, (lm_inds[kk, 1], lm_inds[kk, 0]), radius=0, color=(0, 0, 255), thickness=-1)
    #        cv2.imshow('test', depth)
    #        cv2.waitKey(0)
#
    #        pl.show()
    #else:
    lms3d = None
    lms_depth = None

    return depth, color, cam_pose[:3, :3], cam_pose[:3, 3], K, lms3d, lms_depth, world_space_normals


def get_3d_points(depth, K, Rt, lms, rend_size=(512, 512), mm_scale=False, normals=None):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]

    cx = (960 - K[0][2])
    cy = K[1][2]
    znear = 0.1
    zfar = 2.0
    if mm_scale:
        znear = 200
        zfar = 2000#20 # 2000 #

    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)
    cam_pose = np.eye(4)
    Rt = np.array(Rt)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T @ Rt[:, 3]

    #lm_inds = lms.copy()
    #lm_inds[:, 0] = lms[:, 1] * rend_size[0]
    #lm_inds[:, 1] = lms[:, 0] * rend_size[1]
    #lm_inds = np.floor(lm_inds).astype(np.int32)
    xx, yy = np.meshgrid(np.arange(rend_size[0]), np.arange(rend_size[1]))
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    lm_inds = np.stack([xx, yy], axis=-1).astype(np.int32)
    lms3d, lms_depth, normals = unproject_points(lm_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), cam_pose, normals=normals)

    return lms3d, lms_depth, normals


def unproject_points(ppos, depth, rend_size, K, Rt, normals=None):
    points = np.ones((ppos.shape[0], 4))
    points[:, [1, 0]] = ppos.astype(float)
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    ppos[:, 0] = np.clip(ppos[:, 0], 0, 3840) # 1279) #
    ppos[:, 1] = np.clip(ppos[:, 1], 0, 2160) # 959) #
    points_depth = depth[ppos[:, 0], ppos[:, 1]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    if normals is not None:
        normals = normals[ppos[:, 0], ppos[:, 1], :]
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN
    #pl = pv.Plotter()
    #pl.add_points(points)
    #pl.show()
    if normals is not None:
        pass
        #normals = np.matmul(normals, Rt[:3, :3].T)

    return points, points_depth, normals


def render(E, mesh, out_prefix=None, mm_scale=True, frame_id=None, is_rough=False):
    EX_tmp = E
    EX_tmp = Pose(EX_tmp).invert()


    depth_img, rend_img, R, t, K, lms3d, _, normals = render_glcam(mesh, KK, EX_tmp[:3, :], None,
                                                                   rend_size=(1280, 960), mm_scale=mm_scale)

    return rend_img, depth_img, normals




def project(points, E):
    points_hom = np.ones([points.shape[0], 4])
    points_hom[:, :3] = points
    points_camera_space = (E @ points_hom.T).T[:, :3]
    points_image_plane = (KK @ points_camera_space.T).T
    points_depth = points_image_plane[:, 2]
    points_image = points_image_plane[:, :2] / points_image_plane[:, 2:]
    return points_image, points_depth


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


def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def gen_render_samples(m, N, scale=4):
    m = m.copy()
    m.vertices /= scale
    cams = fibonacci_sphere(N + 2)[1:-1]
    cams.reverse()
    all_points = []
    all_normals = []
    for cam_origin in cams:
        E = m3dLookAt(np.array([-0.00818554, -0.0196824, 0.577607]),
                      np.mean(m.vertices, axis=0),
                      # lms_rough[i][53, :], #np.mean(m.vertices, axis=0),
                      np.array([-0.00713758, 0.999157, 0.0404241]))
        E = m3dLookAt(np.array(cam_origin) * 0.6,
                      np.zeros([3]),
                      # lms_rough[i][53, :], #np.mean(m.vertices, axis=0),
                      np.array([0, 1, 0]))

        rgb, depth, normals = render(E, m, out_prefix=None, mm_scale=False, is_rough=False)

        # cv2.imshow('test', rgb)
        # cv2.waitKey(0)

        EX_tmp = E
        EX_tmp = Pose(EX_tmp).invert()

        points3d, _, normals = get_3d_points(depth, KK, EX_tmp[:3, :], None, rend_size=(1280, 960), mm_scale=False,
                                             normals=normals)

        valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
        points3d = points3d[valid, :]
        normals = normals[valid, :]

        # back face removal
        ray_dir = points3d - np.array(cam_origin) * 0.6
        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
        angle = np.sum(ray_dir * normals, axis=-1)

        all_points.append(points3d[angle < -0.01, :])
        all_normals.append(normals[angle < -0.01, :])
        # pl = pv.Plotter()
        # pl.add_points(points3d[angle < 0, :], scalars=normals[angle < 0, 0])
        # pl.add_points(points3d[angle >= 0, :], color='red')
        # pl.add_points(np.array(cam_origin)*0.6, point_size=20)
        # pl.add_mesh(m)
        # pl.show()
    return np.concatenate(all_points, axis=0)*scale, np.concatenate(all_normals, axis=0)
if __name__ == '__main__':
    N = 20
    m = trimesh.load('/mnt/hdd/NRR_FLAME/andrei/expression_2/warped.ply', process=False)

    all_points, all_normals = gen_render_samples(m, N, scale=1)
    colors = distinctipy.get_colors(N)
    pl = pv.Plotter()
    pl.add_mesh(m)
    #for i, p3d in enumerate(all_points):
        # pl.add_points(p3d, color=colors[i])
    pl.add_points(all_points, scalars=all_normals[:, 0])
    #    print(p3d.shape)
    pl.show()

