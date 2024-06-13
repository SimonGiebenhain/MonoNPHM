import os

import point_cloud_utils as pcu
import numpy as np
import pyvista as pv
import trimesh
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image
#import env_paths
import os.path as osp
import traceback
import tyro
from multiprocessing import Pool

from mononphm.data.manager import DataManager
from mononphm import env_paths


REMOVE_LARGE_FACS = True

def get_color(tex_path, uv_coords):
    img = np.array(Image.open(tex_path))
    imw = img.shape[1]
    imh = img.shape[0]
    u = np.clip((uv_coords[:, 0] * (imw-1)).astype(int), 0, imw-1)
    v = np.clip((uv_coords[:, 1] * (imh-1)).astype(int), 0, imh-1)

    colors = img[imh-1- v, u]
    return colors


def sample_fields(n_samps, n_samps_off, sigma, s, e):
    manager = DataManager(high_res=True)
    mesh = manager.get_raw_mesh(s, e, mesh_type='pcu', textured=True)
    texture_path = osp.join(osp.dirname(manager.get_raw_path(s, e)), mesh.textures[0])

    print('Starting')
    if not OFF_SURFACE_ONLY:
        face_region_mesh = manager.get_registration_mesh(s, e, mesh_type='pcu')

        if REMOVE_LARGE_FACS:
            triangle_areas = pcu.mesh_face_areas(mesh.v, mesh.f)
            edge1 = np.linalg.norm(mesh.v[mesh.f[:, 0], :] - mesh.v[mesh.f[:, 1], :], axis=-1)
            edge2 = np.linalg.norm(mesh.v[mesh.f[:, 1], :] - mesh.v[mesh.f[:, 2], :], axis=-1)
            edge3 = np.linalg.norm(mesh.v[mesh.f[:, 0], :] - mesh.v[mesh.f[:, 2], :], axis=-1)
            edges_all = np.concatenate([edge1, edge2, edge3], axis=0)
            max_edge = np.mean(edges_all) * 2.5
            max_area = np.mean(triangle_areas) * 2.5
            #print(max_area)
            too_big = triangle_areas > max_area
            valid = ~too_big
            too_long = (edge1 > max_edge) | (edge2 > max_edge) | (edge3 > max_edge)
            valid_edge_length = ~too_long
            valid = valid & valid_edge_length

            valid_face_indices = np.arange(mesh.f.shape[0])
            valid_face_indices = valid_face_indices[valid]
            cv, nv, cf, nf = pcu.connected_components(mesh.vertex_data.positions, mesh.face_data.vertex_ids[valid, :])
            biggest_component = np.argmax(nf)
            valid_f = cf == biggest_component
            valid_faces = mesh.face_data.vertex_ids[valid, :][valid_f, :]
            valid_face_indices = valid_face_indices[valid_f]
            print('removed large faces')

        if VISUALIZE and REMOVE_LARGE_FACS:


            m_raw = trimesh.Trimesh(mesh.vertex_data.positions,
                                    valid_faces, process=False)
            m_regi = trimesh.Trimesh(face_region_mesh.vertex_data.positions,
                                    face_region_mesh.face_data.vertex_ids, process=False)

            pl = pv.Plotter()
            pl.add_mesh(m_raw)
            #pl.add_mesh(m_regi, color='red')
            pl.show()


        n = pcu.estimate_mesh_vertex_normals(mesh.vertex_data.positions, mesh.face_data.vertex_ids)


        # Generate random samples on the mesh (v, f, n)cut
        # f_i are the face indices of each sample and bc are barycentric coordinates of the sample within a face
        f_i, bc = pcu.sample_mesh_random(mesh.vertex_data.positions, mesh.face_data.vertex_ids, num_samples=n_samps)

        if REMOVE_LARGE_FACS:
            valid_samples = [np.isin(f_i, valid_face_indices)][0]
            f_i = f_i[valid_samples]
            bc = bc[valid_samples]

        print('sampled mesh')

        # compute field values for points on the surface
        surf_points = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, f_i, bc, mesh.vertex_data.positions)
        surf_normals = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, f_i, bc, n)
        surf_uv_coords = (mesh.face_data.wedge_texcoords[f_i] * bc[:, :, np.newaxis]).sum(1)
        surf_colors = get_color(texture_path, surf_uv_coords)

        above = manager.cut_throat(surf_points, s, e)

        surf_points = surf_points[above, :]
        surf_normals = surf_normals[above, :]
        surf_colors = surf_colors[above, :]

        print('sampled and cut throat')

        # determine which points lie in the facial region
        if face_region_mesh is not None:
            face_region_mesh.vertex_data.positions = face_region_mesh.vertex_data.positions[mask, :]
            non_face_vertices = np.arange(mask.shape[0])[~mask]
            good_faces = ~np.any(np.isin(face_region_mesh.face_data.vertex_ids, non_face_vertices), axis=-1)

            face_region_mesh.face_data.vertex_ids = face_region_mesh.face_data.vertex_ids[good_faces, :]
            (_, corrs_x_to_y, corrs_y_to_x) = pcu.chamfer_distance(surf_points.astype(np.float32),
                                                                   face_region_mesh.vertex_data.positions,
                                                                   return_index=True, p_norm=2) #max_points_per_leaf=250)

            d_region = np.linalg.norm(surf_points.astype(np.float32) - face_region_mesh.vertex_data.positions[corrs_x_to_y, :], axis=-1)
            face_region = d_region < 0.01 #1/25 #5/25
            outer_face_region = (d_region < 5/25) &  np.logical_not(face_region) #0.01 #1/25 #5/25


            print('computed distance to face region mesh')


            if VISUALIZE:
                pl = pv.Plotter()
                #pl.add_mesh(surf_points[face_region, :], scalars=surf_normals[face_region, 0])
                pl.add_mesh(surf_points[face_region, :], scalars=surf_colors[face_region, :], rgb=True)
                #pl.add_mesh(trimesh.Trimesh(face_region_mesh.vertex_data.positions,
                #                                                        face_region_mesh.face_data.vertex_ids, process=False))
                pl.add_points(face_region_mesh.vertex_data.positions)
                pl.show()
    else:
        f_i, bc = pcu.sample_mesh_random(mesh.vertex_data.positions, mesh.face_data.vertex_ids, num_samples=n_samps_off*2)
        n = pcu.estimate_mesh_vertex_normals(mesh.vertex_data.positions, mesh.face_data.vertex_ids)

        # compute field values for points on the surface
        surf_points = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, f_i, bc, mesh.vertex_data.positions)

        above = manager.cut_throat(surf_points, s, e)

        surf_points = surf_points[above, :]


    rnd_idx = np.random.randint(0, surf_points.shape[0], n_samps_off)
    points = surf_points[rnd_idx, :] + np.random.randn(n_samps_off, 3) * sigma

    #sdfs, fi, bc = pcu.signed_distance_to_mesh(points, mesh.vertex_data.positions, mesh.face_data.vertex_ids,)
    d, fi, bc = pcu.closest_points_on_mesh(points, mesh.vertex_data.positions, mesh.face_data.vertex_ids,)

    print('computed nearest neighbors')
    ### Convert barycentric coordinates to 3D positions

    normals = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, fi, bc, n)
    closest_uv = (mesh.face_data.wedge_texcoords[fi] * bc[:, :, np.newaxis]).sum(1)

    colors = get_color(texture_path, closest_uv)

    print('Sampled texture')

    if VISUALIZE:
        if not OFF_SURFACE_ONLY:
            pl = pv.Plotter()
            pl.add_mesh(trimesh.Trimesh(mesh.vertex_data.positions, mesh.face_data.vertex_ids))
            pl.add_points(surf_points, scalars=surf_normals[:, 0])
            #pl.add_points(surf_points, scalars=surf_colors, rgb=True)
            pl.show()

    if OFF_SURFACE_ONLY or face_region_mesh is not None:
        if OFF_SURFACE_ONLY:
            return {
                    'off-surface': {'points': points,
                                    'colors': colors,
                                    'normals': normals,
                                    #'sdfs': sdfs
                                    }
                    }
        else:

            rnd_idx_non_face = np.random.randint(0, np.sum(~face_region), n_samps_off)

            return {'face': {'points':surf_points[face_region, :],
                             'colors': surf_colors[face_region, :],
                             'normals': surf_normals[face_region, :]},
                    'non-face': {'points': surf_points[~face_region, :][rnd_idx_non_face, :],
                                 'colors': surf_colors[~face_region, :][rnd_idx_non_face, :],
                                 'normals': surf_normals[~face_region, :][rnd_idx_non_face, :]},
                    'off-surface': {'points': points,
                                     'colors': colors,
                                     'normals': normals,
                                     #'sdfs': sdfs
                                    },
                    'outer-face': {'points':surf_points[outer_face_region, :],
                                     'colors': surf_colors[outer_face_region, :],
                                     'normals': surf_normals[outer_face_region, :]}
                    }
    else:
        return surf_points, surf_normals #surf_colors, #points, sdfs,  colors, normals


def run_subject(s, expressions=None):
    manager = DataManager(high_res=True)
    if expressions is None:
        expressions = manager.get_expressions(subject=s)
    for e in expressions:
        if e is None:
            continue

        if osp.exists(manager.get_train_path_off_surface(s, e, rnd_file=NUM_SPLITS - 1)) and not VISUALIZE:
            print('SKIPPING:', s, e)
            continue
        else:
            print('Running: ', s, e)
            print(manager.get_train_path_off_surface(s, e, rnd_file=NUM_SPLITS - 1))
        try:
            print(s, e)
            N_SAMPS = 25000000
            N_SAMPS_OFF = 1000000
            if VISUALIZE:
                N_SAMPS = N_SAMPS // 10
                N_SAMPS_OFF = N_SAMPS_OFF // 10
            resultss = [sample_fields(N_SAMPS, N_SAMPS_OFF, sigma=0.05, s=s, e=e) for _ in range(1)]

            print('done with pcu stuff')
            if VISUALIZE:
                print(resultss[0]['off-surface']['points'].shape)

                pl = pv.Plotter()
                if not OFF_SURFACE_ONLY:
                    pl.add_points(resultss[0]['face']['points'])
                    pl.add_points(resultss[0]['non-face']['points'], color='yellow')
                    pl.add_points(resultss[0]['outer-face']['points'], color='purple')
                pl.add_points(resultss[0]['off-surface']['points'])#, scalars=resultss[0]['off-surface']['sdfs'])
                pl.show()

            data_off = np.concatenate([np.concatenate([results['off-surface']['points'],
                                       results['off-surface']['normals'],
                                      results['off-surface']['colors'].astype(np.float32),
                                       #results['off-surface']['sdfs'][:, np.newaxis]
                                                       ], axis=1) for results in resultss], axis=0)

            data_off = data_off.astype(np.float32)
            if not OFF_SURFACE_ONLY:
                data_face = np.concatenate([np.concatenate([results['face']['points'],
                                            results['face']['normals'],
                                            results['face']['colors'].astype(np.float32)
                                            ], axis=1) for results in resultss], axis=0)
                data_face = data_face.astype(np.float32)

                data_non_face = np.concatenate([np.concatenate([results['non-face']['points'],
                                                results['non-face']['normals'],
                                                results['non-face']['colors'].astype(np.float32)
                                                ], axis=1) for results in resultss], axis=0)
                data_non_face = data_non_face.astype(np.float32)
                data_outer_face = np.concatenate([np.concatenate([results['outer-face']['points'],
                                                                results['outer-face']['normals'],
                                                                results['outer-face']['colors'].astype(np.float32)
                                                                ], axis=1) for results in resultss], axis=0)
                data_outer_face = data_outer_face.astype(np.float32)

            if not VISUALIZE:
                out_dir_s = manager.get_train_dir(s)
                os.makedirs(out_dir_s, exist_ok=True)
                #print(out_dir_s, e)
                if not OFF_SURFACE_ONLY:
                    chunks_face = np.array_split(data_face, NUM_SPLITS, axis=0 )
                    chunks_non_face = np.array_split(data_non_face, NUM_SPLITS, axis=0 )
                    chunks_outer = np.array_split(data_outer_face, NUM_SPLITS, axis=0 )
                chunks_off = np.array_split(data_off, NUM_SPLITS, axis=0 )
                if not OFF_SURFACE_ONLY:
                    for i, chunk_face in enumerate(chunks_face):
                        np.save(manager.get_train_path_face_region(s, e, rnd_file=i), chunk_face)
                    for i, chunk_non_face in enumerate(chunks_non_face):
                        np.save(manager.get_train_path_non_face_region(s, e, rnd_file=i), chunk_non_face)
                        for i, chunk_outer in enumerate(chunks_outer):
                            np.save(manager.get_train_path_static_front_region(s, e, rnd_file=i), chunk_outer)
                for i, chunk_off in enumerate(chunks_off):
                    np.save(manager.get_train_path_off_surface(s, e, rnd_file=i), chunk_off)

                print('done with saving')
        except Exception as ex:
            print('EXCEPTION', s, e)
            print(traceback.format_exc())


def main(starti : int, endi : int = None):
    '''
    Preprocess data that is used for training supervision.
    Args:
        starti: Process identities above this (or equal) ID
        endi: and below this ID

    '''
    print(f'STARINTG PROCESING FROM INDEX {starti} to INDEX {endi}')
    manager = DataManager()

    all_subjects = manager.get_all_subjects()
    print(len(all_subjects))

    print(f"FOUND {len(all_subjects)} subjects!")


    out_dir = env_paths.SUPERVISION_IDENTITY
    os.makedirs(out_dir, exist_ok=True)

    if endi is None:
        all_subjects = [s for s in all_subjects if s >= starti]
    else:
        all_subjects = [s for s in all_subjects if s >= starti and s < endi]
    print(all_subjects)


    if not VISUALIZE:
        p = Pool(4)
        p.map(run_subject, all_subjects)
        p.close()
        p.join()

    else:
        # only perform for first subject
        run_subject(all_subjects[0])


VISUALIZE = False # Set this to True for debugging, or to understand better how the training supervision looks like

OFF_SURFACE_ONLY = False
face_region_template = pcu.load_triangle_mesh(env_paths.ASSETS + '/template_face_up.ply')
mask = np.load(env_paths.ASSETS + 'face.npy')

NUM_SPLITS = env_paths.NUM_SPLITS

if __name__ == '__main__':
    tyro.cli(main)