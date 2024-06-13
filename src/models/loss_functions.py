import torch


def compute_loss(batch, decoder, latent_codes, epoch,
                 color_only: bool = False,
                 face_only: bool = False,
                 loss_fn_percep=None,
                 device=None
                 ):
    if 'path' in batch:
        del batch['path']

    if (batch.get('subj_ind') > latent_codes.codebook['geo'].embedding.weight.data.shape[0]).any():
        print('hi')
    if (batch.get('app_ind') > latent_codes.codebook['app'].embedding.weight.data.shape[0]).any():
        print('hi')
    if 'exp' in latent_codes.codebook and (
        batch.get('idx') > latent_codes.codebook['exp'].embedding.weight.data.shape[0]).any():
        print('hi')

    # push to gpu
    batch_cuda_nphm = {k: v.float().to(device) for (k, v) in zip(batch.keys(), batch.values())}
    idx = {'geo': batch.get('subj_ind').to(device),
           'app': batch.get('app_ind').to(device),
           'exp': batch.get('idx').to(device)}

    # obtain geo, expr and app latent codes
    cond = latent_codes(idx)

    # compute all losses
    loss_dict = loss_joint(batch_cuda_nphm, decoder, cond, epoch)

    return loss_dict


'''
 geo loss
 corresp loss
 color loss

'''


def compute_loss_corresp(points_posed, points_neutral, decoder, cond, anchors, has_corresp):
    ex_decoder_dict = {'queries': points_posed,
                       'cond': cond,
                       'anchors': anchors}
    out_delta = decoder.ex_model(ex_decoder_dict)
    pred_can = points_posed + out_delta['offsets']
    loss_corresp = (pred_can[has_corresp, ...] - points_neutral[has_corresp, ...]).square().mean()
    return loss_corresp


def compute_def_reg(decoder, cond, out_surface, out_outer, out_off, is_neutral):
    zero_loss = torch.zeros([1], device=cond['geo'].device).mean()
    if not decoder.ex_model.neutral_only:
        # enforce deformation field to be zero elsewhere
        nsamps = 50

        samps = (torch.rand(cond['geo'].shape[0], nsamps, 3, device=cond['geo'].device) - 0.5) * 2.5
        ex_decoder_dict = {'queries': samps,
                           'cond': cond,
                           'anchors': decoder.id_model.get_anchors(cond['geo'])}

        out_delta_reg = decoder.ex_model(ex_decoder_dict)

        loss_reg_zero = out_delta_reg['offsets'].square().mean()
    else:
        loss_reg_zero = zero_loss
        out_delta_reg = zero_loss

    # usrface, outer, off, far, delta, delta_reg
    # for neutral expressions, encourage small deformations

    if not decoder.ex_model.neutral_only and (is_neutral.sum() > 0):

        def_list = [out_surface['offsets'][is_neutral, ...].view(-1)]
        if out_outer is not None:
            def_list.append(out_outer['offsets'].reshape(-1))
        if out_off is not None:
            def_list.append(out_off['offsets'].reshape(-1))
        loss_neutral_def = torch.cat(def_list).square().mean()
    else:
        loss_neutral_def = zero_loss

    return loss_reg_zero, loss_neutral_def, out_delta_reg


def compute_lat_reg(decoder, cond):
    lat_reg_shape = (torch.norm(cond['geo'], dim=-1) ** 2).mean()
    if 'exp' in cond:
        lat_reg_expr = (torch.norm(cond['exp'], dim=-1) ** 2).mean()
    else:
        lat_reg_expr = torch.zeros_like(lat_reg_shape)
    if 'app' in cond:
        lat_reg_app = (torch.norm(cond['app'], dim=-1) ** 2).mean()
    else:
        lat_reg_app = torch.zeros_like(lat_reg_shape)

    if hasattr(decoder.id_model, 'lat_dim_glob'):
        symm_dist, middle_dist = decoder.id_model.get_symm_reg(cond['geo'].squeeze(1), cond_type='geo')
    else:
        symm_dist = torch.zeros_like(lat_reg_shape)
        middle_dist = torch.zeros_like(lat_reg_shape)

    if hasattr(decoder.id_model, 'lat_dim_glob') & decoder.id_model.color_branch:
        symm_dist_app, middle_dist_app = decoder.id_model.get_symm_reg(cond['app'].squeeze(1), cond_type='app')
    else:
        symm_dist_app = torch.zeros_like(lat_reg_shape)
        middle_dist_app = torch.zeros_like(lat_reg_shape)

    return lat_reg_shape, lat_reg_app, lat_reg_expr, symm_dist.mean(), symm_dist_app.mean(), middle_dist.mean(), middle_dist_app.mean()


def compute_color_loss(out_surface, out_outer, out_off, hair_mask, batch_cuda):
    loss_color_surface = (batch_cuda['color_surface'] - out_surface['color']).abs()
    color_losses = [loss_color_surface.view(-1)]

    color_loss_outer = (
        batch_cuda['color_surface_outer'] - out_outer['color']).abs()
    color_losses.append(color_loss_outer.view(-1))

    if torch.sum(hair_mask) > 0:
        color_loss_off = (batch_cuda['color_off_surface'][hair_mask, ...] - out_off['color'][
            hair_mask, ...]).abs() / 5  # TODO: hardcoded hyper-param.
        color_losses.append(color_loss_off.view(-1))

    return torch.cat(color_losses).mean()


def compute_geometry_loss(out_surface, out_outer, out_off, out_far, batch_cuda, hair_mask):
    gradient_surface = out_surface['gradient']

    sdf_losses = []
    normal_losses = []
    eikonal_losses = []

    # points on surface should have SDF==0
    surf_sdf_loss = torch.abs(out_surface['sdf']).squeeze(dim=-1)
    sdf_losses.append(surf_sdf_loss.view(-1))

    # enforce that gradient of SDF point into the direction of the normals
    surf_normal_loss = (gradient_surface - batch_cuda['normals_surface']).norm(2, dim=-1)
    normal_losses.append(surf_normal_loss.view(-1))

    # Eikonal contraint
    surf_grad_loss = torch.abs(gradient_surface.norm(dim=-1) - 1)
    eikonal_losses.append(surf_grad_loss.view(-1))

    # on surface, back of head
    if torch.sum(hair_mask) > 0:
        surf_sdf_loss_outer = torch.abs(out_outer['sdf']).squeeze(dim=-1)
        surf_normal_loss_outer = torch.clamp((out_outer['gradient'][hair_mask, ...] -
                                              batch_cuda['normals_surface_outer'][hair_mask, ...]).norm(2, dim=-1),
                                             None,
                                             0.75) / 4
        sdf_losses.append(surf_sdf_loss_outer.view(-1))
        normal_losses.append(surf_normal_loss_outer.view(-1))

    surf_grad_loss_outer = torch.abs(out_outer['gradient'].norm(dim=-1) - 1)
    eikonal_losses.append(surf_grad_loss_outer.view(-1))

    # off surface
    gradient_off = out_off['gradient']
    surf_grad_loss_off = torch.abs(gradient_off.norm(dim=-1) - 1)
    eikonal_losses.append(surf_grad_loss_off.view(-1))

    # off surface, canonical space only
    space_sdf_loss = torch.exp(-1e1 * torch.abs(out_far['sdf'])).mean()
    space_grad_loss_far = torch.abs(out_far['gradient'].norm(dim=-1) - 1)
    eikonal_losses.append(space_grad_loss_far.view(-1))

    # combine individual losses
    tot_sdf_loss = torch.cat(sdf_losses,
                             dim=0).mean()
    tot_normal_loss = torch.cat(normal_losses, dim=0).mean()
    tot_eikonal_loss = torch.cat(eikonal_losses, dim=0).mean()

    return tot_sdf_loss, tot_normal_loss, tot_eikonal_loss, space_sdf_loss


def compute_hyper_loss(out_surface, out_outer, out_off, out_far, out_delta_reg, is_neutral):
    all_hyper_coords = [out_surface['hyper_coords'],
                        out_far['hyper_coords'],
                        out_delta_reg['hyper_coords']]
    all_hyper_loss = torch.cat(all_hyper_coords, dim=1).abs().mean()
    # all_hyper_loss = torch.cat(all_hyper_coords, dim=1).square().mean()

    if (is_neutral).sum() > 0:
        hyper_coords_neutral = [out_surface['hyper_coords'][is_neutral, ...].reshape(-1), ]
        if out_outer is not None:
            hyper_coords_neutral.append(out_outer['hyper_coords'].reshape(-1), )
        if out_off is not None:
            hyper_coords_neutral.append(out_off['hyper_coords'].reshape(-1))
        # hyper_loss_neutral = torch.cat(hyper_coords_neutral, dim=0).square().mean()
        hyper_loss_neutral = torch.cat(hyper_coords_neutral, dim=0).abs().mean()
        all_hyper_loss += 10 * hyper_loss_neutral  # TODO: ATTENTION random hyper-param
    return all_hyper_loss


def loss_joint(batch_cuda, decoder, cond, epoch):
    b = batch_cuda['is_neutral'].shape[0]
    assert b > 1  # otherwise squeeze() will mess uo things
    # for certain types of points we compute certain loss, precompute the necessary masks
    is_neutral = batch_cuda['is_neutral'].squeeze(dim=-1) == 1
    has_corresp = batch_cuda['has_corresp'].squeeze(dim=-1) == 1
    has_anchors = batch_cuda['has_anchors'].squeeze(dim=-1) == 1
    hair_mask = batch_cuda['supervise_hair'].squeeze(dim=-1) == 1

    # concatenate all types of points --> we only need one network call --> efficiency
    all_points = torch.cat([
        batch_cuda['points_surface'],
        batch_cuda['points_surface_outer'],
        batch_cuda['points_off_surface'],
        batch_cuda['sup_grad_far']
    ], dim=1)
    # needed to extract different types of points after network call
    n_points_face = batch_cuda['points_surface'].shape[1]
    n_points_outer = batch_cuda['points_surface_outer'].shape[1]
    # n_points_off_face = batch_cuda['points_off_surface'].shape[1]
    n_points_far = batch_cuda['sup_grad_far'].shape[1]

    # MAIN COMPUTATION
    out_network = decoder({'queries': all_points}, cond, return_grad=True)

    output_modalities = ['sdf', 'gradient', 'offsets']
    if decoder.id_model.color_branch:
        output_modalities.append('color')
    if 'hyper_coords' in out_network.keys():
        output_modalities.append('hyper_coords')

    # split output according to different types of query points
    out_surface = {k: out_network[k][:, :n_points_face, :] for k in output_modalities}
    out_outer = {k: out_network[k][:, n_points_face:n_points_face + n_points_outer, :] for k in output_modalities}
    out_off = {k: out_network[k][:, n_points_face + n_points_outer:-n_points_far, :] for k in output_modalities}
    out_far = {k: out_network[k][:, -n_points_far:, :] for k in output_modalities}

    # compute different types of losses; encapsulated into individual functions for readability
    tot_sdf_loss, tot_normal_loss, tot_grad_loss, space_sdf_loss = compute_geometry_loss(out_surface, out_outer,
                                                                                         out_off, out_far, batch_cuda,
                                                                                         hair_mask)

    if decoder.id_model.color_branch:
        color_loss = compute_color_loss(out_surface, out_outer, out_off, hair_mask, batch_cuda)

    # latent regularizers
    lat_reg_shape, lat_reg_app, lat_reg_expr, symm_dist, symm_dist_app, middle_dist, middle_dist_app = compute_lat_reg(
        decoder, cond)

    # loss_anchors
    if out_network['anchors'] is not None and has_anchors.sum() > 0:
        loss_anchors = (
                out_network['anchors'][has_anchors, ...] - batch_cuda['gt_anchors'][has_anchors, ...]).square().mean()
    else:
        loss_anchors = torch.zeros_like(tot_grad_loss)

    # correspondences
    if epoch < 3000 and not decoder.ex_model.neutral_only and (has_corresp).sum() > 0:
        loss_corresp = compute_loss_corresp(batch_cuda['corresp_posed'],
                                            batch_cuda['corresp_neutral'],
                                            decoder, cond, out_network.get('anchors', None),
                                            has_corresp)
        if epoch > 750:
            loss_corresp *= 0.25
    else:
        loss_corresp = torch.zeros_like(tot_grad_loss)

    # deformation regulariazation
    loss_reg_zero, loss_neutral_def, out_delta_reg = compute_def_reg(decoder, cond, out_surface, out_outer, out_off,
                                                                     is_neutral)

    loss_dict = {
        'surf_sdf': tot_sdf_loss,
        'normals': tot_normal_loss,
        'space_sdf': space_sdf_loss,
        'eikonal': tot_grad_loss,
        'reg_shape': lat_reg_shape,
        'reg_expr': lat_reg_expr,
        'anchors': loss_anchors,
        'symm_dist': symm_dist,
        'middle_dist': middle_dist,
        'corresp': loss_corresp,
        'loss_reg_zero': loss_reg_zero,
        'loss_neutral_zero': loss_neutral_def,
    }

    if decoder.id_model.color_branch:
        color_loss_dict = {
            'color': color_loss,
            'reg_app': lat_reg_app.mean(),
            'symm_dist_app': symm_dist_app.mean(),
            'middle_dist_app': middle_dist_app.mean(),
        }

        loss_dict.update(color_loss_dict)

    if decoder.ex_model.n_hyper > 0:
        hyper_loss = compute_hyper_loss(out_surface, out_outer, out_off, out_far, out_delta_reg, is_neutral)
        loss_dict.update({'hyper': hyper_loss})

    return loss_dict