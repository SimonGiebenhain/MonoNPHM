# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)


def intrinsics2projection(K, znear, zfar, width, height):
    x0 = 0
    y0 = 0
    return np.array([
    [2 * K[0, 0] / width, -2 * K[0, 1] / width, (width - 2 * K[0, 2] + 2 * x0) / width, 0],
    [0, -2 * K[1, 1] / height, (height - 2 * K[1, 2] + 2 * y0) / height, 0],
    [0, 0, (-zfar - znear) / (zfar - znear), -2 * zfar * znear / (zfar - znear)],
    [0, 0, -1, 0]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4)
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)

#----------------------------------------------------------------------------



from PIL import Image
import numpy as np


def backproject(depth_maps, normal_maps, Ks, Es, rgb=None, masks=None):
    points3d = {}
    normals3d = {}
    rgb3d = {}
    for cam_id in depth_maps.keys():
        depth_map = depth_maps[cam_id]
        normal_map = normal_maps[cam_id]


        ys = np.arange(depth_map.shape[0])
        xs = np.arange(depth_map.shape[1])
        p_screen = np.dstack(np.meshgrid(xs, ys, [1])).reshape((-1, 3))
        depth_mask = (depth_map > 0)  & (depth_map < 1.4)
        if masks is not None:
            # upsample mask
            I = Image.fromarray(masks[cam_id])
            I = I.resize((I.size[0]*2, I.size[1]*2))
            depth_mask = np.logical_and(depth_mask, np.array(I).astype(np.bool))
        depths = depth_map[depth_mask]
        p_screen = p_screen[depth_mask.reshape(-1)]
        p_screen_canonical = p_screen @ Ks[cam_id].invert().T
        p_cam = p_screen_canonical * np.expand_dims(depths, 1)
        p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
        p_world = p_cam_hom @ Es[cam_id].T
        ns = np.ones_like(p_world)
        ns[:, :3] = normal_map[depth_mask]
        n_world = ns @ Es[cam_id].T

        points3d[cam_id] = p_world[:, :3]
        normals3d[cam_id] = n_world[:, :3]
        if rgb is not None:
            rgb_lin = rgb[cam_id].reshape((-1, 3))
            rgb_valid = rgb_lin[depth_mask.reshape(-1)]
            rgb3d[cam_id] = rgb_valid

    if rgb is None:
        return points3d, normals3d
    else:
        return points3d, normals3d, rgb3d


def get_view_dirs(Ks, Es, image_shape, rgb=None, masks=None):
    points3d = {}
    view_dirs = {}
    for cam_id in Ks.keys():
        ys = np.arange(image_shape[0])
        xs = np.arange(image_shape[1])
        p_screen = np.dstack(np.meshgrid(xs, ys, [1])).reshape((-1, 3))
        if masks is not None:
            # upsample mask
            I = Image.fromarray(masks[cam_id])
            I = I.resize((I.size[0]*2, I.size[1]*2))
            depth_mask = np.logical_and(depth_mask, np.array(I).astype(np.bool))
        p_screen = np.reshape(p_screen, [-1, 3])
        p_screen_canonical = p_screen @ Ks[cam_id].invert().T
        p_cam = p_screen_canonical * 1
        p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
        p_world = p_cam_hom @ Es[cam_id].T

        points3d[cam_id] = p_world[:, :3]

        origin = Es[cam_id][:3, 3]
        view_dirs[cam_id] = p_world[:, :3] - origin
        view_dirs[cam_id] /= np.linalg.norm(view_dirs[cam_id], axis=-1, keepdims=True)

        #if rgb is not None:
        #    rgb_lin = rgb[cam_id].reshape((-1, 3))
        #    rgb_valid = rgb_lin[depth_mask.reshape(-1)]
        #    rgb3d[cam_id] = rgb_valid

    #if rgb is None:
    #    return points3d, normals3d
    #else:

    return view_dirs





    return


import os
import numpy as np
import cv2

# borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None and (uvcoords is None or uvfaces is None):
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 0], faces[i, 1], faces[i, 2]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            if texture is not None:
                with open(mtl_name, 'w') as f:
                    f.write('newmtl %s\n' % material_name)
                    s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                    f.write(s)

                    if normal_map is not None:
                        name, _ = os.path.splitext(obj_name)
                        normal_name = f'{name}_normals.png'
                        f.write(f'disp {normal_name}')
                        # out_normal_map = normal_map / (np.linalg.norm(
                        #     normal_map, axis=-1, keepdims=True) + 1e-9)
                        # out_normal_map = (out_normal_map + 1) * 0.5

                        cv2.imwrite(
                            normal_name,
                            # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                            normal_map
                        )
                cv2.imwrite(texture_name, texture)


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    colors = []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)

            if len(tokens) > 4:
                if '.' in tokens[4]:
                    color = [int(float(x)*255) for x in tokens[4:7]]
                else:
                    color = [int(x) for x in tokens[4:7]]
                if len(color) != 3:
                    msg = "Color %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                colors.append(color)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts).astype(np.float32)
    uvcoords  = np.array(uvcoords ).astype(np.float32)
    colors = np.array(colors).astype(int)
    faces  = np.array(faces).astype(np.int64); faces = faces.reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces).astype(np.int64); uv_faces = uv_faces.reshape(-1, 3) - 1

    return verts, uvcoords, colors, faces, uv_faces


def m3dLookAt(eye, target, up):
    eye = eye.astype(float)
    target = target.astype(float)
    up = up.astype(float)
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



if __name__ == '__main__':
    p_id = 18
    seq_name = 'EXP-5-mouth'
    frame = 0
    folder = f'/mnt/rohan/cluster/doriath/tkirschstein/data/famudy/full/{p_id:03d}/sequences/{seq_name}/annotations/tracking/NPHM/'
    mesh_path = f"{folder}/remeshed_{frame:05d}debugOLD_local_new.obj"
    verts, uvcoords, colors, faces, uv_faces = load_obj(mesh_path)
    print('hi')