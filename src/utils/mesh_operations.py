import numpy as np
import point_cloud_utils as pcu

def cut_trimesh_vertex_mask(mesh, mask):
    invalid_mask = np.logical_not(mask)
    invalid_faces = mesh.vertex_faces[invalid_mask]
    invalid_faces = np.unique(invalid_faces.reshape([-1]))
    invalid_faces = invalid_faces[invalid_faces >= 0]
    invalid_faces_mask = np.zeros(dtype=bool, shape=[mesh.faces.shape[0]])
    invalid_faces_mask[invalid_faces] = 1
    mesh.update_faces(np.logical_not(invalid_faces_mask))
    mesh.remove_unreferenced_vertices()
    return mesh


def remove_large_triangles(mesh, max_size_factor = 2.5):

    triangle_areas = pcu.mesh_face_areas(mesh.v, mesh.f)
    edge1 = np.linalg.norm(mesh.v[mesh.f[:, 0], :] - mesh.v[mesh.f[:, 1], :], axis=-1)
    edge2 = np.linalg.norm(mesh.v[mesh.f[:, 1], :] - mesh.v[mesh.f[:, 2], :], axis=-1)
    edge3 = np.linalg.norm(mesh.v[mesh.f[:, 0], :] - mesh.v[mesh.f[:, 2], :], axis=-1)
    edges_all = np.concatenate([edge1, edge2, edge3], axis=0)
    max_edge = np.mean(edges_all) * max_size_factor
    max_area = np.mean(triangle_areas) * max_size_factor
    # print(max_area)
    too_big = triangle_areas > max_area
    valid = ~too_big
    too_long = (edge1 > max_edge) | (edge2 > max_edge) | (edge3 > max_edge)
    valid_edge_length = ~too_long
    valid = valid & valid_edge_length

    valid_face_indices = np.arange(mesh.f.shape[0])
    valid_face_indices = valid_face_indices[valid]
    cv, nv, cf, nf = pcu.connected_components(mesh.v, mesh.f[valid, :])
    biggest_component = np.argmax(nf)
    valid_f = cf == biggest_component
    valid_faces = mesh.f[valid, :][valid_f, :]
    valid_face_indices = valid_face_indices[valid_f]
    return valid_face_indices


def furthest_point_sampling(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds], sample_inds


