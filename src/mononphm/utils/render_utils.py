import os
import tyro
import numpy as np
import torch
import pathlib
import trimesh
import math

# Disable antialiasing:
import OpenGL.GL

suppress_multisampling = True
old_gl_enable = OpenGL.GL.glEnable

def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)

OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample

def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)

OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample
import pyrender






KK = np.array([
    [2440, 0, 480],
    [0, 2440, 640],
    [0, 0, 1]], dtype=np.float32)

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
                 rend_size=(512, 512),
                 znear=0.1,
                 zfar=2.0,
                 render_normals = True):

    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()

    if hasattr(mesh.visual, 'material'):
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
    else:
        glossiness = 1.0
        if isinstance(glossiness, list):
            glossiness = float(glossiness[0])
        roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            baseColorFactor=[255, 255, 255, 255],
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material)

    #pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material)

    # Scene creation
    scene = pyrender.Scene(ambient_light = [0.45,0.45,0.45, 1.0])

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)

    scene.add(cam, pose=Rt)

    # Set up the light
    instensity = 0.75

    light1 = pyrender.PointLight(intensity=instensity)
    light2 = pyrender.PointLight(intensity=instensity)

    light_pose1 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 300]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))
    light_pose2 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 0]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))

    light_pose1[:3, 3] = Rt[:3, 3]/2 + np.array([0.15, 0.1, -0.15])
    light_pose2[:3, 3] = Rt[:3, 3]/2 + np.array([-0.15, 0.1, -0.15])


    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)


    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   )
    if render_normals:
        r._renderer._program_cache = CustomShaderCache()

    normals_or_color, depth = r.render(scene,
                                       flags=pyrender.constants.RenderFlags.FLAT)
    #I = Image.fromarray(normals_or_color)
    #I.show()
    r.delete()
    if render_normals:
        world_space_normals = normals_or_color / 255 * 2 - 1

    depth[depth == 0] = float('inf')
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)

    if render_normals:
        return depth, world_space_normals
    else:
        return depth, normals_or_color


def get_3d_points(depth, K, Rt, rend_size=(512, 512), normals=None, znear=0.1, zfar=2.0):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)


    xx, yy = np.meshgrid(np.arange(rend_size[0]), np.arange(rend_size[1]))
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    pixel_inds = np.stack([xx, yy], axis=-1).astype(np.int32)
    lms3d = unproject_points(pixel_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), Rt)

    return lms3d


def unproject_points(ppos, depth, rend_size, K, Rt):
    points = np.ones((ppos.shape[0], 4))
    points[:, [1, 0]] = ppos.astype(float)
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    ppos[:, 0] = np.clip(ppos[:, 0], 0, rend_size[0])
    ppos[:, 1] = np.clip(ppos[:, 1], 0, rend_size[1])
    points_depth = depth[ppos[:, 0], ppos[:, 1]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN

    return points

def project_points(points, intrinsics, world_to_cam_pose):
    p_world = np.hstack([points, np.ones((points.shape[0], 1))])
    p_cam = p_world @ world_to_cam_pose.T
    depths = p_cam[:, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_screen = p_cam[:, :3] @ intrinsics.T
    p_screen[:, 2] = np.squeeze(depths, 1)  # Return depth as third coordinate
    return p_screen

def project_points_torch(points, intrinsics, world_to_cam_pose):
    p_world = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=-1)
    p_cam = p_world @ world_to_cam_pose.T
    depths = p_cam[:, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_cam[:, 0] *= -1 # since the NeuS rendering is happening in OpenGL we need to correct it as such

    p_screen = p_cam[:, :3] @ intrinsics.T
    p_screen[:, 2] = depths.squeeze(dim=1)  # Return depth as third coordinate
    return p_screen

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


def render_and_backproject(m,
                           down_scale_factor : int = 2,
                           crop : int = 50):
    E = m3dLookAt(np.array([0, 0, 1]) * 0.6,
                  np.zeros([3]),
                  np.array([0, 1, 0]))

    rend_size = (1280 // down_scale_factor, 960 // down_scale_factor)
    crop = crop // down_scale_factor

    KK = np.array(
        [[2440 / down_scale_factor, 0.00000000e+00, (rend_size[1] / 2) - crop],
         [0.00000000e+00, 2440 / down_scale_factor, (rend_size[0] / 2) - crop],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )
    rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)

    m.vertices /= 4
    depth, rgb = render_glcam(m, KK, E, rend_size=rend_size, render_normals=False)
    points3d = get_3d_points(depth, KK, E, rend_size=rend_size)
    points3d *= 4
    m.vertices *= 4

    #valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
    #points3d = points3d[valid, :]

    #normals = np.transpose(normals, [1, 0, 2])
    #normals = normals.reshape([-1, 3])
    #normals = normals[valid, :]
    return rgb, points3d

def gen_render_samples(m, N, scale=4, down_scale_factor=1, crop=0, render_color=False, return_grid=False):
    m = m.copy()
    m.vertices /= scale

    rend_size = (1280 // down_scale_factor, 960 // down_scale_factor)
    crop = crop // down_scale_factor

    KK = np.array(
        [[2440 / down_scale_factor, 0.00000000e+00, (rend_size[1] / 2) - crop],
         [0.00000000e+00, 2440 / down_scale_factor, (rend_size[0] / 2) - crop],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
    )
    rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)




    cams = fibonacci_sphere(N + 2)[1:-1]
    cams.reverse()
    if render_color or return_grid:
        eyes = np.array(fibonacci_sphere(1000))
        eyes = eyes[eyes[:, 2] > 0.5, :]
        eyes = eyes[eyes[:, 1] < 0.7, :]
        eyes = eyes[eyes[:, 1] > -0.7, :]
        #eyes = eyes[eyes[:, 2] > 0.6, :]
        #eyes = eyes[eyes[:, 1] < 0.55, :]
        #eyes = eyes[eyes[:, 1] > -0.55, :]
        cams = []
        for i in range(N):
            if i == 0:
                cams.append(np.array([0, 0, 1]))
            else:
                rnd_indx = np.random.randint(0, len(eyes))
                #rnd_indx = 10
                cams.append(eyes[rnd_indx])

    all_points = []
    all_normals = []

    for cam_origin in cams:

        if N == 1:
            cam_origin = [0, 0, 1]
        E = m3dLookAt(np.array(cam_origin) * 0.6,
                      np.zeros([3]),
                      np.array([0, 1, 0]))

        depth, normals = render_glcam(m, KK, E, rend_size=rend_size, render_normals=not render_color)
        if not render_color:
            n = normals + 1
            n /= 2.0
            n *= 255.0
            n = n.astype(np.uint8)
            #I = Image.fromarray(n)
            #I.show()
        points3d = get_3d_points(depth, KK, E, rend_size=rend_size)

        if not render_color and not return_grid:
            valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
            points3d = points3d[valid, :]

            normals = np.transpose(normals, [1, 0, 2])
            normals = normals.reshape([-1, 3])
            normals = normals[valid, :]
            # back face removal
            ray_dir = points3d - np.array(cam_origin) * 0.6
            ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
            angle = np.sum(ray_dir * normals, axis=-1)

            all_points.append(points3d[angle < -0.01, :])
            all_normals.append(normals[angle < -0.01, :])
        else:
            if not render_color:

                ray_dir = points3d - np.array(cam_origin) * 0.6
                ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
                ray_dir = ray_dir.reshape([normals.shape[1], normals.shape[0], 3])
                ray_dir = np.transpose(ray_dir, [1, 0, 2])

                #point_img = ray_dir
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()
#
                #point_img = normals
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()
#
#
                angle = np.sum(ray_dir * normals, axis=-1)
                #point_img = angle
                #point_img -= np.nanmin(point_img)
                #point_img /= np.nanmax(point_img)
                #point_img *= 255
                #point_img = point_img.astype(np.uint8)
                #point_img[np.isnan(point_img)] = 255
                #I = Image.fromarray(point_img)
                #I.show()


                valid_angle = angle < -0.01
                valid_angle[np.isnan(angle)] = False
                #valid_angle = valid_angle.reshape([normals.shape[1], normals.shape[0], 3])
                #valid_angle = np.transpose(valid_angle, [1, 0, 2])




            point_img = points3d.reshape([normals.shape[1], normals.shape[0], 3])
            point_img = np.transpose(point_img, [1, 0, 2])

            #if not render_color:
            #    point_img[np.logical_not(valid_angle)] = np.NaN
            #    normals[np.logical_not(valid_angle)] = np.NaN

            #point_img -= np.nanmin(point_img)
            #point_img /= np.nanmax(point_img)
            #point_img *= 255
            #point_img = point_img.astype(np.uint8)
            #point_img[np.isnan(point_img)] = 255
            #I = Image.fromarray(point_img)
            #I.show()
            #In = Image.fromarray(normals)
            #In.show()

            #p = np.reshape(point_img, [-1, 3])
            #pl = pv.Plotter()
            #pl.add_points(p)
            #pl.show()



            all_points.append(point_img)
            all_normals.append(normals)

    if render_color or return_grid:
        return [points*scale for points in all_points],\
                all_normals
    else:
        return np.concatenate(all_points, axis=0)*scale,\
               np.concatenate(all_normals, axis=0)


class SimpleMesh():
    def __init__(self, v, f):
        self.v = v
        self.f = f




