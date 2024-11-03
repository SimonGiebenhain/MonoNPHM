import pyvista as pv
import dreifus.pyvista
import trimesh
import numpy as np
from typing import Optional
from dreifus.matrix import Pose, Intrinsics
from math import ceil
from pyvista import numpy_to_texture
import torch
def project_torch(points: torch.Tensor, world_to_cam_pose, intrinsics) -> np.ndarray:
    """
    Projects 3D points onto the image plane defined by the camera pose and intrinsics.

    Parameters
    ----------
        points: 3D points [N, 3]
        pose: Camera pose. Can be either CAM_2_WORLD or WORLD_2_CAM
        intrinsics: Intrinsics of the camera

    Returns
    -------
        Projected points [N, 3] in image space where the third coordinate defines the depth from the camera space.
        Note that no occlusions are handled (difficult for point clouds) and points that would be outside the image
        won't be filtered.
    """

    p_world = torch.cat([points, torch.ones(list(points.shape[:-1]) + [1], device=points.device)], dim=-1)
    p_cam = torch.matmul(p_world, world_to_cam_pose.permute(0, 2, 1))
    depths = p_cam[:, :, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_screen = torch.matmul(p_cam[:, :, :3], intrinsics.permute(0, 2, 1))
    return p_screen[..., :2]


def render(pl : Optional[pv.Plotter],
           pred_mesh : Optional[trimesh.Trimesh],
           c2w_extrinsics : Pose,
           intrinsics : Intrinsics,
           vertex_colors = None,
           no_light = False,
           pointcloud : Optional[np.ndarray] = None,
           pointcloud_scalars : Optional[np.ndarray] = None,
           clim=[0, 0.01],
           wh = None,
           light_pose_zero = False,
           points = None,
           point_colors = None,
           ):
    if pl is None:
        if wh is None:
            w, h = ceil(intrinsics.cx * 2) + 1, int(intrinsics.cy * 2) + 1
            if abs(w-1100) < 5:
                w = 1100
            if abs(h-1604) < 5:
                h = 1604
        else:
            w = wh[0]
            h = wh[1]
        pl = pv.Plotter(window_size=(w, h),
                        off_screen=True,
                        lighting='none'
                        )
        if pred_mesh is not None:


            if vertex_colors is None:
                if hasattr(pred_mesh.visual, 'uv'):
                    pv_pred_mesh = pv.wrap(pred_mesh)
                    pred_mesh.active_t_coords = pred_mesh.visual.uv
                    tex = numpy_to_texture(np.array(pred_mesh.visual.material.image).astype(np.uint8))
                    pl.add_mesh(pv_pred_mesh, texture=tex)
                else:
                    pl.add_mesh(pred_mesh)
            else:
                pl.add_mesh(pred_mesh, scalars=vertex_colors, rgb=True)

        if pointcloud is not None:
            if pointcloud_scalars is not None:
                pl.add_mesh(pointcloud, scalars=pointcloud_scalars, clim=clim)
                #pl.add_mesh(pointcloud, scalars=pointcloud_scalars, rgb = True)
            else:
                pl.add_mesh(pointcloud, color='red')

        if points is not None:
            pl.add_points(points, scalars=point_colors, rgb=True, render_points_as_spheres=True, point_size=10)

    light_pos = np.zeros([1, 4])

    if not light_pose_zero:
        light_pos[0, 3] = 1
        light_pos[0, 0] = 0.4
        light_pos[0, 1] = -0.4

        light_pos = (c2w_extrinsics @ light_pos.T).T
        light_pos = light_pos[:, :3]
        light = pv.Light(position=(light_pos[0, 0], light_pos[0, 1], light_pos[0, 2]),
                         show_actor=False, positional=True,
                         cone_angle=90, exponent=5, intensity=0.6)
    else:
        light_pos[0, 3] = -1
        light = pv.Light(position=(light_pos[0, 0], light_pos[0, 1], light_pos[0, 2]), focal_point = (0, 0, 1),
                         show_actor=False, positional=True,
                         cone_angle=90, exponent=5, intensity=1.0)


    if not no_light:
        pl.add_light(light)

    image = dreifus.pyvista.render_from_camera(pl, c2w_extrinsics, intrinsics)[..., :3]
    pl.close()
    return image