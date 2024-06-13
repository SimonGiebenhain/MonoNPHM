import torch
import json, os, yaml
import torch
torch.set_float32_matmul_precision('high')

import numpy as np
import tyro
from typing import Any, List, Tuple, Union, Literal, Optional

import tempfile

from mononphm.models.training import TrainerAutoDecoder
from mononphm import env_paths


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def subprocess_fn(rank, CFG, args, temp_dir):

    # Init torch.distributed.
    if CFG['num_gpus'] > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=CFG['num_gpus'])
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=CFG['num_gpus'])

    # Init torch_utils.

    trainer = TrainerAutoDecoder(rank=rank, cfg=CFG, args=args, **args)
    trainer.train_model(30001)


def main(cfg_file : str,
         exp_name : str,
         color_branch : bool = False,
         pass_pos : bool = False,
         include_outer_mlp : bool = False,
         anchors_path : Optional[str] = None,
         neutral_only :bool = False,
         lambda_sdf_corrective : float = 0,
         is_monolith : bool = False,
         no_validation : bool = False,
         color_only : bool = False,
         face_only : bool = False,
         mvs_dataset : bool = False,
         model_type : Literal['nphm', 'global', 'grid', 'triplane', 'latent-mesh', 'eg3d'] = 'nphm',
         use_patches : bool = False,
         debug_run : bool = False,
         pass_exp2app : bool = False,
         old_global : bool = False,
         legacy_nphm : bool = False,
         omit_extra_mlp : bool = False,
         modulation_in_communication : bool = False,
         uv_communication : bool = False,
         off_only : bool = False,
         global_color : bool = False,
         variational : bool = False,
         disable_color_communication : bool = False,
         path_to_latent_id : Optional[str] = None,
         ):

    ignore_outer_mlp = not include_outer_mlp

    CFG = yaml.safe_load(open(cfg_file, 'r'))


    if color_only:
        assert color_branch
    if lambda_sdf_corrective > 0:
        CFG['training']['lambdas']['sdf_corrective'] = lambda_sdf_corrective

    exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(exp_name)
    fname = exp_dir + 'configs.yaml'
    if not os.path.exists(exp_dir):
        print('Creating checkpoint dir: ' + exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
        with open(fname, 'w') as yaml_file:
            yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
    else:
        with open(fname, 'r') as f:
            print('Loading config file from: ' + fname)
            CFG = yaml.safe_load(f)

    print(json.dumps(CFG, sort_keys=True, indent=4))


    #name for wandb
    project = 'mononphm'
    if debug_run:
        project += '_debug'
        CFG['training']['ckpt_interval'] = 5
    CFG['project_name'] = project
    CFG['num_gpus'] = torch.cuda.device_count()
    CFG['exp_name'] = exp_name

    print(f'FOUND {CFG["num_gpus"]} GPUs')


    kwargs = EasyDict(num_gpus=torch.cuda.device_count(),
                      exp_dir=exp_dir,
                      model_type=model_type,
                      include_app=color_branch,
                      neutral_only=neutral_only,
                      )

    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if CFG['num_gpus'] == 1:
            subprocess_fn(rank=0, CFG=CFG, args=kwargs, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(CFG, kwargs, temp_dir), nprocs=CFG['num_gpus'])



if __name__ ==  '__main__':
    tyro.cli(main)