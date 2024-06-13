import numpy as np
import torch

# data
from mononphm import env_paths
from mononphm.data.face_dataset import NPHMdataset

# NNs
from mononphm.models.canonical_space import get_id_model
from mononphm.models.deepSDF import GlobalFieldNew
from mononphm.models.deformations import DeformationNetwork
from mononphm.models.base import LatentCodes






def set_up_datasets(cfg,
                     \
                     lm_inds = None,
                     debug_run : bool = False,
                     neutral_only : bool = False,
                     no_validation : bool = False,
                     use_patches : bool = False,
                     uv_communication : bool = False,
                    **kwards,
                    ):
    DataSetClass = NPHMdataset

    if debug_run:
        cfg['training']['batch_size'] = 10
        cfg['training']['npoints_face'] = 100
        cfg['training']['npatches_per_batch'] = 2
        cfg['training']['ckpt_interval'] = 100

    train_dataset = DataSetClass(
                                 mode='train',
                                      n_supervision_points_corresp=cfg['training'].get('npoints_corresp', 250),
                                      n_supervision_points_face=cfg['training'].get('npoints_face', 1000),
                                      n_supervision_points_non_face=cfg['training'].get('npoints_non_face', 250),
                                      n_supervision_points_off_surface=cfg['training'].get('npoints_off_surface', 250),
                                      batch_size=cfg['training']['batch_size'],
                                      # sigma_near=CFG['training']['sigma_near'], #TODO probably should use that?!
                                      lm_inds=lm_inds,
                                      # is_closed=args.closed
                                      neutral_only=neutral_only,
                                      no_validation=no_validation,
                                      num_anchors=cfg['decoder']['id'].get('nloc'), #, 39),
                                      num_symm=cfg['decoder']['id'].get('nsymm_pairs'), #, 16),
                                      )
    val_dataset = DataSetClass(mode='val',
                                   n_supervision_points_corresp=cfg['training'].get('npoints_corresp', 250),
                                   n_supervision_points_face=cfg['training'].get('npoints_face', 1000),
                                   n_supervision_points_non_face=cfg['training'].get('npoints_non_face', 250),
                                   n_supervision_points_off_surface=cfg['training'].get('npoints_off_surface', 250),
                                    batch_size=cfg['training']['batch_size'],
                                    # sigma_near=CFG['training']['sigma_near'], #TODO probably should use that?!
                                    lm_inds=lm_inds,
                                    # is_closed=args.closed
                                    neutral_only=neutral_only,
                                    no_validation=no_validation,
                                    num_anchors=cfg['decoder'].get('decoder_nloc', 39),
                                   num_symm=cfg['decoder'].get('nsymm_pairs', 16),
                               )

    return train_dataset, val_dataset


def set_up_networks(cfg,
                    model_type,
                    rank=None,
                    anchors = None,
                    **kwargs,
                    ):

    if model_type == 'nphm':

        id_decoder = get_id_model(cfg['decoder'],
                                  3 + cfg['decoder']['ex']['nhyper'],
                                  include_color_branch=True,
                                  rank=rank,
                                  )
        ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=cfg['decoder']['ex']['lat_dim_id'],
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim_glob'],
                                            lat_dim_loc_shape=cfg['decoder']['id']['lat_dim_loc_geo'],
                                            n_loc=cfg['decoder']['id']['nloc'],
                                            anchors=anchors,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=False,
                                            n_hyper=cfg['decoder']['ex']['nhyper'],
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )


    elif model_type == 'global':

        id_decoder = GlobalFieldNew(
        lat_dim=cfg['decoder']['id']['lat_dim'],
        lat_dim_app=cfg['decoder']['id']['lat_dim_app'],
        hidden_dim=cfg['decoder']['id']['hidden_dim'],
        nlayers=cfg['decoder']['id']['nlayers'],
        nlayers_color=cfg['decoder']['id'].get('nlayers_color', 6),
        out_dim=1,
        input_dim=3 + cfg['decoder']['ex']['nhyper'],
        color_branch=True,
        num_freq_bands=cfg['decoder']['id'].get('nfreq_bands_geo', 0),
        freq_exp_base=cfg['decoder']['id'].get('freq_base_geo', 0.5),
        lat_dim_exp=cfg['decoder']['ex']['lat_dim_ex'],
        num_freq_bands_color=cfg['decoder']['id'].get('nfreq_bands_color', 0),
        freq_exp_base_color=cfg['decoder']['id'].get('freq_base_color', 2.0),
        is_monolith=False,
        communication_dim=0,
        uv_communication=False,
            include_anchors=True,
            anchors=anchors,
        )
        ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=-1,
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim'],
                                            lat_dim_loc_shape=-1,
                                            n_loc=-1,  # CFG['decoder']['id']['nloc'],
                                            anchors=None,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=False,
                                            n_hyper=cfg['decoder']['ex']['nhyper'],
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )
    else:
        raise ValueError(f'Unknown model type {model_type}')

    return id_decoder, ex_decoder

def set_up_codes(cfg,
                 model_type,
                 id_decoder,
                 ex_decoder,
                 train_dataset,
                 val_dataset,
                 \
                 neutral_only : bool = False,
                 include_app : bool = False,
                 variational : bool = False,
                 codebook_numbers_train = None,
                 codebook_numbers_val = None,
                 **kwargs,
                 ):
    # Initializing latent codes.
    modalities = ['geo']
    types = ['vector']
    if model_type == 'nphm':
        n_channels = [id_decoder.lat_dim_glob_geo + id_decoder.lat_dim_loc_geo * (id_decoder.n_anchors + 1)] # [id_decoder.lat_dim]
    elif model_type == 'global':
        n_channels = [cfg['decoder']['id']['lat_dim']]
    elif model_type == 'grid':
        n_channels = [id_decoder.lat_dim * id_decoder.resolution ** 3]
    elif model_type == 'triplane':
        n_channels = [id_decoder.lat_dim * id_decoder.resolution ** 2 * 3]
    elif model_type == 'eg3d':
        n_channels = [id_decoder.lat_dim_app]  # TODO currently generating geo and app together
    elif model_type == 'latent-mesh':
        assert 1 == 2
        if False:
            n_channels = [id_decoder.lat_dim * id_decoder.n_points]
        else:
            n_channels = [id_decoder.lat_dim * uv_res ** 2 * (1 + uv_layers_positive + uv_layers_negative)]

    n_train_geo = len(train_dataset.subjects) if train_dataset is not None else 381
    n_val_geo = len(val_dataset.subjects) if val_dataset is not None else 10
    if codebook_numbers_train is not None:
        n_train_geo = codebook_numbers_train['geo']
    n_latents_train = [n_train_geo]#269] #237]
    n_latents_val = [n_val_geo] #2

    if not neutral_only:
        modalities.append('exp')
        types.append('vector')
        n_channels.append(ex_decoder.lat_dim_expr)
        n_train_exp = len(train_dataset) if train_dataset is not None else 7707
        n_val_exp = len(val_dataset) if train_dataset is not None else 224
        if codebook_numbers_train is not None:
            n_train_exp = codebook_numbers_train['exp']
            n_val_exp = codebook_numbers_val['exp']
        n_latents_train.append(n_train_exp) #5646) #4905)
        n_latents_val.append(n_val_exp) #46 )
    if include_app:
        modalities.append('app')
        types.append('vector')
        if model_type == 'nphm':
            n_channels.append(id_decoder.lat_dim_glob_app + id_decoder.lat_dim_loc_app * (id_decoder.n_anchors + 1))
        elif model_type == 'grid':
            n_channels.append(id_decoder.lat_dim_app * id_decoder.resolution ** 3)
        elif model_type == 'triplane':
            n_channels.append(id_decoder.lat_dim_app * id_decoder.resolution ** 2 * 3)
        elif model_type == 'eg3d':
            n_channels.append(id_decoder.lat_dim_app)
        elif model_type == 'global':
            n_channels.append(id_decoder.lat_dim_app)
        elif model_type == 'latent-mesh':
            if False:
                n_channels.append(id_decoder.lat_dim_app * id_decoder.n_points)
            else:
                n_channels.append(
                    id_decoder.lat_dim_app * uv_res ** 2 * (1 + uv_layers_positive + uv_layers_negative))

        n_train_app = n_train_geo
        n_val_app = n_val_geo
        if codebook_numbers_train is not None:
            n_train_app = codebook_numbers_train['app']
            n_val_app = codebook_numbers_val['app']
        n_latents_train.append(n_train_app) #269) #237 )
        n_latents_val.append(n_val_app) #2)
        if train_dataset is not None and train_dataset.MIRROR:
            n_latents_train = [n*2 for n in n_latents_train]
            n_latents_val = [n*2 for n in n_latents_val]

    latent_codes = LatentCodes(n_latents=n_latents_train,
                                    n_channels=n_channels,  # CFG['decoder']['ex']['lat_dim_ex']],
                                    modalities=modalities,
                                    types=types,
                                    init_zeros=True,
                               variational=variational,
                                    )
    latent_codes_val = LatentCodes(n_latents=n_latents_val,
                                        n_channels=n_channels,  # CFG['decoder']['ex']['lat_dim_ex']],
                                        modalities=modalities,
                                        types=types,
                                        init_zeros=True,
                                   variational=variational,
                                        )

    return latent_codes, latent_codes_val


def set_up_all(cfg,
               args,
               skip_dataset : bool = False,
               rank=None,
               codebook_numbers_train = None,
               codebook_numbers_val = None,
               skip_codebooks : bool = False,
               ):

        # Load stuff required for NPHM.
        lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH.format(cfg['decoder'].get('decoder_nloc', 65)))
        anchors_path = env_paths.ANCHOR_MEAN_PATH.format(cfg['decoder'].get('decoder_nloc', 65))

        anchors = torch.from_numpy(np.load(anchors_path)).float().unsqueeze(0).unsqueeze(0)

        if not skip_dataset:
            train_dataset, val_dataset = set_up_datasets(cfg,
                                                         lm_inds=lm_inds,
                                                         **args
                                                         )
        else:
            train_dataset = None
            val_dataset = None

        id_decoder, ex_decoder = set_up_networks(cfg,
                                                 anchors=anchors,
                                                 rank=rank,
                                                 **args
                                                 )

        if not skip_codebooks:
            latent_codes, latent_codes_val = set_up_codes(cfg,
                                                      args.model_type,
                                                      id_decoder,
                                                      ex_decoder,
                                                      train_dataset,
                                                      val_dataset,
                                                      include_app=args.include_app,
                                                      neutral_only=args.neutral_only,
                                                      codebook_numbers_train=codebook_numbers_train,
                                                      codebook_numbers_val=codebook_numbers_val,
                                                      )
        else:
            latent_codes = None
            latent_codes_val = None


        if not skip_dataset:
            print(f'Train Dataset has {len(train_dataset.subjects)} Subjects and {len(train_dataset.subject_IDs)} Expressions')
            print(f'Val Dataset has {len(val_dataset.subjects)} Subjects and {len(val_dataset.subject_IDs)} Expressions')

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'id_decoder': id_decoder,
            'ex_decoder': ex_decoder,
            'latent_codes': latent_codes,
            'latent_codes_val': latent_codes_val,
        }






