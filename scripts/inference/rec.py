import json, os, yaml
import torch
import numpy as np
import tyro
from typing import Literal

from mononphm.photometric_tracking.tracking import track
from mononphm.photometric_tracking.wrapper import WrapMonoNPHM
from mononphm.models.neural3dmm import nn3dmm
from mononphm.models import setup_training
from mononphm import env_paths
from mononphm.utils.others import EasyDict




def inverse_rendering(net, seq_name, expressions, n_expr, out_dir=None,
                      is_stage2 : bool = False,
                      is_global=False,
                      intrinsics_provided : bool = True,
                      is_video : bool = True,
                      downsample_factor : float = 1/6,
                      ):

    if not is_stage2:
        exp_dir = f'{out_dir}/stage1/{seq_name}/{expressions[0]:05d}//'

        while os.path.exists(f'{exp_dir}/z_geo.npy'):

            expressions[0] += 1

            exp_dir = f'{out_dir}/stage1/{seq_name}/{expressions[0]:05d}//'

        wrapped_net = WrapMonoNPHM(net)

        cfg = f'{env_paths.CODE_BASE}/scripts/configs/tracking/stage1_ffhq.yaml'

        if is_video:
            cfg = f'{env_paths.CODE_BASE}/scripts/configs/tracking/stage1_kinect.yaml'
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)

        if is_global:
            cfg['opt']['lambdas_reg']['reg_global_app'] *= 10
            cfg['opt']['lambdas_reg']['reg_global_geo'] *= 10


        if expressions[0] == 0:
            is_first_frame = True
        else:
            is_first_frame = False


        if is_first_frame and is_video:
            cfg['opt']['lambda_reg_expr'] = 1000
        elif not intrinsics_provided and not is_video:
            cfg['opt']['lambda_reg_expr'] = 250

        track(wrapped_net, cfg, seq_name, expressions,
              out_dir=out_dir,
              fix_id=not is_first_frame,
              lr_scale=0.99 if not is_first_frame else 1,
              seq_name=seq_name,
              intrinsics_provided=intrinsics_provided,
              #is_video=is_video,
              downsampling_factor=downsample_factor,
              )
    else:
        cfg = f'{env_paths.CODE_BASE}/scripts/configs/tracking/stage2_ffhq.yaml'

        if is_video:
            cfg = f'{env_paths.CODE_BASE}/scripts/configs/tracking/stage2_kinect.yaml'
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)

        out_dir_stage1 = out_dir + '/stage1/'
        out_dir = out_dir #+ '/stage2'
        os.makedirs(out_dir, exist_ok=True)

        wrapped_net = WrapMonoNPHM(net)

        track(wrapped_net, cfg, seq_name, expressions, out_dir=out_dir,
              out_dir_stage1=out_dir_stage1,
              fix_id=True,
              lr_scale=0.5,
              fine_tune_id=True,
              seq_name=seq_name,
              intrinsics_provided=intrinsics_provided,
              #is_video=is_video,
              downsampling_factor=downsample_factor,
              )


def main(seq_name : str,
         model_type : Literal['nphm', 'global'],
         exp_name: str,
         ckpt: int,
         intrinsics_provided : bool = True,
         is_video : bool = True,
         is_stage2 : bool = False,
         downsample_factor : float = 1/6,
    ):


    EXP_NAME = exp_name
    CKPT = ckpt

    out_base = env_paths.TRACKING_OUTPUT

    CFG = yaml.safe_load(open(f'{env_paths.EXPERIMENT_DIR}/{EXP_NAME}/configs.yaml', 'r'))

    exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(EXP_NAME)


    print(json.dumps(CFG, sort_keys=True, indent=4))


    CFG['project_name'] = 'tracking'
    CFG['num_gpus'] = torch.cuda.device_count()
    CFG['exp_name'] = EXP_NAME

    print(f'FOUND {CFG["num_gpus"]} GPUs')

    kwargs = EasyDict(num_gpus=torch.cuda.device_count(),
                      exp_dir=exp_dir,
                      model_type=model_type,
                      )

    rank = 0
    num_gpus = 0
    random_seed = 42
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)

    model_dir = f'{env_paths.EXPERIMENT_DIR}/{EXP_NAME}'
    path = f'{model_dir}/checkpoints/checkpoint_epoch_{CKPT}.tar'
    checkpoint = torch.load(path)

    prepped_entities = setup_training.set_up_all(CFG, kwargs,
                                                 skip_dataset=True,
                                                 skip_codebooks=True)

    prepped_entities['id_decoder'].load_state_dict(checkpoint['id_decoder'])
    prepped_entities['ex_decoder'].load_state_dict(checkpoint['ex_decoder'])
    id_decoder = prepped_entities['id_decoder'].cuda()
    ex_decoder = prepped_entities['ex_decoder'].cuda()


    n3dmm = nn3dmm(id_model=id_decoder,
                   ex_model=ex_decoder,
                   expr_direction='backward')


    out_dir = f'{out_base}/{EXP_NAME}/'
    os.makedirs(out_dir, exist_ok=True)


    # read number of frames
    if intrinsics_provided or is_video:
        files = os.listdir(f'{env_paths.DATA_TRACKING}/{seq_name}/source/')

        n_expr = len(files)
    else:
        n_expr = 1



    expressions = range(0, n_expr, 1)
    if is_stage2:
        expressions = range(0, n_expr, 1)

    if is_stage2:
        assert is_video
        inverse_rendering(n3dmm, seq_name, expressions, n_expr,
                          out_dir=out_dir,
                          is_stage2=is_stage2,
                          is_global=model_type == 'global',
                          intrinsics_provided=intrinsics_provided,
                          is_video=is_video, downsample_factor=downsample_factor,
                          )
    else:

        for e, expression in enumerate(expressions):
            _expressions = [expression]
            inverse_rendering(n3dmm, seq_name, _expressions, n_expr,
                              out_dir=out_dir,
                              is_global=model_type == 'global',
                              intrinsics_provided=intrinsics_provided,
                              is_video=is_video, downsample_factor=downsample_factor,
                              )


if __name__ ==  '__main__':
    tyro.cli(main)