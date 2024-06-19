import os.path
from time import time
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.ops import knn_points

def get_logits(decoder, encoding, grid_points, nbatch_points=100000, return_anchors=False, approx_verts=None, cached_sdfs=None,
               cached_deformations = None,
               return_deformations = False,
               ignore_deformations = False):
    sample_points = grid_points

    # mask out queries that are far away from approximate surface, e.g. from previsou frame
    if approx_verts is not None:
        time_prep = time()
        assert  cached_sdfs is not None

        approx_verts = torch.from_numpy(approx_verts).unsqueeze(0).float()
        verts_split = torch.split(approx_verts, 150000, dim=1)
        def_list = []
        for verts in verts_split:
            in_dict = {'queries': verts.to(encoding['geo'].device), 'cond': encoding}
            if hasattr(decoder.id_model, 'mlp_pos') and decoder.id_model.mlp_pos is not None and 'anchors' not in in_dict:
                in_dict.update({'anchors': decoder.id_model.get_anchors(encoding['geo'])})
            out_ex = decoder.ex_model(in_dict)
            def_list.append(out_ex['offsets'])
        deformation = torch.cat(def_list, dim=1)
        time_knn = time()
        knns = knn_points(sample_points, approx_verts.to(encoding['geo'].device), K=1)
        dists = knns.dists.squeeze()



        velocity = (deformation - torch.from_numpy(cached_deformations).float().to(encoding['geo'].device)).norm(dim=-1).squeeze()

        delta = velocity/2 + 0.00001
        delta = delta[knns.idx.squeeze()]
        #import pyvista as pv
        #pl = pv.Plotter()
        #pl.add_points(approx_verts.detach().squeeze().cpu().numpy(), color='red', point_size=20, render_points_as_spheres=True)
        ##pl.add_points(sample_points.squeeze().detach().cpu().numpy())
        #pl.add_points(sample_points[:, dists < delta, :].squeeze().detach().cpu().numpy(), scalars=delta[dists < delta].detach().cpu().numpy(), point_size=2)
        #pl.show()

        sample_points = sample_points[:, dists < delta, :]
        print(f'KNN took {time()-time_knn} seconds')


        print('Percentage of evaluated MC queries: ', (dists < delta).sum()/torch.numel(dists))
        print(f'It took {time() - time_prep} seconds')
    else:
        deformation = None

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            out_dict = decoder({'queries': points.to(encoding['geo'].device)}, encoding, skip_color=True, ignore_deformations=ignore_deformations)
            logits = out_dict['sdf'].squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())


    logits = torch.cat(logits_list, dim=0).numpy()

    if cached_sdfs is not None:
        cached_sdfs[(dists < delta).squeeze().detach().cpu().numpy()] = logits
        logits = cached_sdfs
    if return_anchors:
        return logits, out_dict['anchors']
    elif return_deformations:
        return logits, deformation.detach().cpu().numpy() if deformation is not None else None
    else:
        return logits


def get_normals(decoder, encoding, grid_points, nbatch_points=100000, return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        out_dict = decoder({'queries': points.to(encoding['geo'].device)}, encoding, return_grad=True)
        grads = out_dict['gradient'].squeeze()
        logits_list.append(grads.squeeze(0).detach().cpu())
        torch.cuda.empty_cache()

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits, out_dict['anchors']
    else:
        return logits

def get_logits_old(decoder, encoding, grid_points, nbatch_points=100000, return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            logits, anchors, color = decoder(points, encoding, None)
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits, anchors
    else:
        return logits


def get_vertex_color(decoder, encoding, vertices, nbatch_points=100000, return_anchors=False, uniform_scaling=False, device=None):
    sample_points = vertices.clone()
    if device is None:
        device = encoding['geo'].device
    print('total_points', sample_points.shape)
    if len(sample_points.shape) == 2:
        sample_points = sample_points.unsqueeze(0)
    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    color_list = []
    for points in grid_points_split:
        with torch.no_grad():
            #logits, anchors, color = decoder(points, encoding, None)
            #print('get vert color', points.shape)
            if points.shape[1] == 0 or points.shape[1] == 0:
                continue
            color = decoder({'queries': points.to(device)}, encoding, None)['color']
            color = color.squeeze()
            color_list.append(color.squeeze(0).detach().cpu())
        torch.cuda.empty_cache()

    color = torch.cat(color_list, dim=0).numpy()
    if uniform_scaling:
        color += 1
        color /= 2
        color *= 255
    else:
        color *= np.array([62.06349782, 52.41366313, 48.37649288])
        color += np.array([109.23604821, 98.02477547, 87.84371274])

    color = np.clip(color, 0, 255)
    color = color.astype(np.uint8)

    #trim.visual = trimesh.visual.ColorVisuals(trim, vertex_colors=vc)
    if return_anchors:
        return color, anchors
    else:
        return color

def get_canonical_vertices(decoder, encoding, vertices, nbatch_points=100000, return_anchors=False, uniform_scaling=False, device=None):
    sample_points = vertices.clone()
    if device is None:
        device = encoding['geo'].device
    print('total_points', sample_points.shape)
    if len(sample_points.shape) == 2:
        sample_points = sample_points.unsqueeze(0)
    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    can_pos_list = []
    for points in grid_points_split:
        with torch.no_grad():
            #logits, anchors, color = decoder(points, encoding, None)
            #print('get vert color', points.shape)
            if points.shape[1] == 0 or points.shape[1] == 0:
                continue
            out_dict = decoder({'queries': points.to(encoding['geo'].device)}, encoding, return_can=True)

            can_positions = out_dict['queries_can'].squeeze()
            can_pos_list.append(can_positions.squeeze(0).detach().cpu())
        torch.cuda.empty_cache()

    can_pos = torch.cat(can_pos_list, dim=0).numpy()

    return can_pos

def get_image_color(decoder, sample_points, encoding, rend_size, uniform_scaling=False, batch_points=None,
                    is_precomputed=False, fitting_dir = None, device = None):
    if device is None:
        device = decoder.device
    split_dim = 0
    if len(sample_points.shape) == 3:
        split_dim = 1
    if batch_points is None:
        batch_points = 25000 if not os.path.exists('/mnt/rohan/') else 5000

    final_rendering = torch.zeros([rend_size[1] * rend_size[0], 3], dtype=torch.uint8)
    grid_points_split = torch.split(sample_points, batch_points, dim=split_dim)
    logits_list = []
    for _points in grid_points_split:
        with torch.no_grad():
            points = _points[:, ~torch.isnan(_points[0, :, 0]), :]
            pred_color_full = torch.zeros(_points.shape, dtype=torch.uint8, device=device)
            if points.shape[1] > 0:
                enc = encoding #.repeat(1, points.shape[1], 1)
                #_, _, pred_color = decoder(points, enc, None)
                pred_color = decoder({'queries': points.to(device)}, enc, None)['color']
                pred_color = pred_color.squeeze()
                if uniform_scaling:
                    pred_color += 1
                    pred_color /= 2
                    pred_color *= 255
                else:
                    pred_color *= torch.tensor([62.06349782, 52.41366313, 48.37649288])
                    pred_color += torch.tensor([109.23604821, 98.02477547, 87.84371274])
                pred_color_full[:, ~torch.isnan(_points[0, :, 0]), :] = pred_color.byte()
            logits_list.append(pred_color_full.detach().cpu().squeeze())
        torch.cuda.empty_cache()


    color = torch.cat(logits_list, dim=0)
    #print(color.shape)
    if fitting_dir is not None:
        final_rendering[~background, :] = color
        color = final_rendering
        color[background, :] = 0
    if is_precomputed:
        color = torch.reshape(color, [rend_size[1], rend_size[0], 3])
        color = color.detach().cpu().numpy()
        #color = np.flip(color, axis=0)
        pred_img = Image.fromarray(color)
    else:
        color = torch.reshape(color, [rend_size[0], rend_size[1], 3])
        color = color.permute(1, 0, 2)
        pred_img = Image.fromarray(color.detach().cpu().numpy())
    return pred_img


def get_logits_backward(decoder_shape, decoder_expr, encoding_shape, encoding_expr, grid_points, nbatch_points=100000, return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            # deform points
            if encoding_expr is not None:
                offsets, _ = decoder_expr(points, encoding_expr.repeat(1, points.shape[1], 1), None)
                points_can = points + offsets
            else:
                points_can = points
            # query geometry in canonical space
            logits, anchors = decoder_shape(points_can, encoding_shape.repeat(1, points.shape[1], 1), None)
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits, anchors
    else:
        return logits


def deform_mesh(mesh, deformer, lat_rep, anchors, lat_rep_shape=None, transform_normals=False):
    points_neutral = torch.from_numpy(np.array(mesh.vertices)).float().unsqueeze(0).to(lat_rep.device)
    normals_neutral = torch.from_numpy(np.array(mesh.vertex_normals)).float().unsqueeze(0).to(lat_rep.device)
    #if decoder.input_dim == 3:
    #    points_neutral = points_neutral[:, :, :3]

    # print(cond.shape)
    if True:
    #with torch.no_grad():
        grid_points_split = torch.split(points_neutral, 5000, dim=1)
        grid_normals_split = torch.split(normals_neutral, 5000, dim=1)
        delta_list = []
        normal_list = []
        for split_id, points in enumerate(grid_points_split):
            if lat_rep_shape is None:
                glob_cond = lat_rep.repeat(1, points.shape[1], 1)
            else:
                glob_cond = torch.cat([lat_rep_shape, lat_rep], dim=-1)
                glob_cond = glob_cond.repeat(1, points.shape[1], 1)
            if anchors is not None:
                d, _ = deformer(points, glob_cond, anchors.unsqueeze(1).repeat(1, points.shape[1], 1, 1))
            else:
                d, _ = deformer(points, glob_cond, None)
            p_prime = points + d
            #if anchors is not None:
            #    grad_inv = jac(deformer, points, glob_cond,
            #                   anchors.unsqueeze(1).repeat(1, points.shape[1], 1, 1)).inverse()
            #else:
            #    grad_inv = jac(deformer, points, glob_cond,
            #                   None).inverse()
            delta_list.append(d.detach().clone())
            #normal_list.append( torch.nn.functional.normalize(
            #    torch.einsum("bnij,bnj->bni", grad_inv.permute(0, 1, 3, 2), grid_normals_split[split_id]).detach().clone().squeeze(), dim=-1)
            #)
            torch.cuda.empty_cache()
        delta = torch.cat(delta_list, dim=1)
        #normals_deformed = torch.cat(normal_list, dim=0)

    pred_posed = points_neutral[:, :, :3] + delta.squeeze()
    # print(torch.any(torch.isnan(pred_posed)))

    verts = pred_posed.detach().cpu().squeeze().numpy()

    #glob_cond = torch.cat([lat_rep_shape, lat_rep], dim=-1)




    mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

    #pl = pv.Plotter(shape=(1,3))
    #pl.subplot(0, 0)

    #pl.add_mesh(mesh, scalars=mesh.vertex_normals[:, 1])

    #pl.subplot(0, 1)
    #pl.add_mesh(mesh_deformed, scalars=normals_deformed[:, 1].detach().cpu().numpy())

    #pl.subplot(0, 2)
    #pl.add_mesh(mesh_deformed, scalars=mesh_deformed.vertex_normals[:,1])

    #pl.link_views()
    #pl.show()
    #pl.close()

    return mesh_deformed

