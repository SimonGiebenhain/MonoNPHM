from __future__ import division
import torch
torch.set_float32_matmul_precision('high')
import torch.optim as optim
from typing import Optional
import json

import os
from glob import glob
import numpy as np
import wandb
import trimesh
import traceback
from PIL import Image

from mononphm.models import setup_training
from mononphm.data.manager import DataManager
from mononphm.models.loss_functions import compute_loss
from mononphm.models.reconstruction import get_image_color, get_logits, get_vertex_color
from mononphm.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from mononphm.utils.render_utils import render_and_backproject
from mononphm.utils.mesh_operations import cut_trimesh_vertex_mask
from mononphm.models.neural3dmm import nn3dmm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrainerAutoDecoder(object):
    def __init__(self,
                 rank,
                 cfg,
                 args,
                 num_gpus,
                 model_type,
                 exp_dir,
                 \
                 include_app: bool = False,
                 color_only: bool = False,
                 face_only: bool = False,
                 use_patches: bool = False,
                 debug_run: bool = False,
                 neutral_only: bool = False,
                 overfit_id: Optional[int] = None,
                 pass_exp2app : bool = False,
                 old_global : bool = False,
                 omit_extra_mlp : bool = False,
                 modulation_in_communication : bool = False,
                 ignore_outer_mlp : bool = True,
                 pass_pos : bool = True,
                 is_monolith : bool = False,
                 uv_communication : bool = False,
                 off_only : bool = False,
                 global_color: bool = False,
                 variational : bool = False,
                 disable_color_communication : bool = False,
                 path_to_latent_id = None,
                 disable_wandb : bool = False,
                 ):

        self.MIRROR = False # TODO ugly, should be coupled with FLAG in dataset class
        prepped_entities = setup_training.set_up_all(cfg, args, rank=rank)


        #num_gpus = args.num_gpus
        exp_dir = args.exp_dir

        neutral_only=args.neutral_only
        self.neutral_only = neutral_only


        # Initialize.
        random_seed = 0
        device = torch.device('cuda', rank)
        np.random.seed(random_seed * num_gpus + rank)
        torch.manual_seed(random_seed * num_gpus + rank)
        torch.backends.cudnn.benchmark = True #cudnn_benchmark  # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = False #allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = False #allow_tf32  # Allow PyTorch to internally use tf32 for convolutions

        print(f'hi from gpu {rank}')

        self.rank = rank
        self.num_gpus = num_gpus
        self.device = device
        self.cfg = cfg['training']


        self.manager = DataManager()
        self.include_app = include_app
        self.include_exp = not neutral_only



        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        # Construct train and validation dataset.
        # -------------------------------------------------------------------------------------------------------------


        self.train_dataset = prepped_entities['train_dataset']
        self.val_dataset = prepped_entities['val_dataset']

        if num_gpus > 1:
            self.train_sampler = torch.utils.data.DistributedSampler(dataset=self.train_dataset,
                                                                 num_replicas=num_gpus,
                                                                 rank=self.rank,
                                                                 shuffle=True,
                                                                 )
            self.val_sampler = torch.utils.data.DistributedSampler(dataset=self.val_dataset,
                                                                     num_replicas=num_gpus,
                                                                     rank=self.rank,
                                                                     shuffle=True,
                                                                     )
        else:
            self.train_sampler = None
            self.val_sampler = None

        num_workers = 2 #TODO: remove hard coded
        self.train_dataset_loader = torch.utils.data.DataLoader(
            self.train_dataset, sampler= self.train_sampler,
            batch_size=self.train_dataset.batch_size, num_workers=num_workers, shuffle=(self.train_sampler is None), pin_memory=True,
            prefetch_factor=2, persistent_workers=num_workers > 0)

        self.val_dataset_loader = torch.utils.data.DataLoader(
            self.val_dataset, sampler=self.val_sampler,
            batch_size=self.val_dataset.batch_size, num_workers=num_workers, shuffle=(self.val_sampler is None), pin_memory=True,
            prefetch_factor=2, persistent_workers=num_workers > 0)


        if self.rank == 0:
            print(f'TrainDataset has  {len(self.train_dataset.subjects)} Identities!')
            print(f'ValDataset has  {len(self.val_dataset.subjects)} Identities!')

        if self.rank == 0:
            with open(f'{exp_dir}/subject_train_index.json', 'w') as f:
                json.dump([int(s) for s in self.train_dataset.subjects], f)
            with open(f'{exp_dir}/expression_train_index.json', 'w') as f:
                json.dump([int(s) for s in self.train_dataset.expression_steps], f)
            with open(f'{exp_dir}/subject_train_pids.json', 'w') as f:
                json.dump([int(s) for s in self.train_dataset.subject_IDs], f)
            with open(f'{exp_dir}/subject_train_inds.json', 'w') as f:
                json.dump([int(s) for s in self.train_dataset.subject_training_ordering], f)
            with open(f'{exp_dir}/subject_val_index.json', 'w') as f:
                json.dump([int(s) for s in self.val_dataset.subjects], f)
            with open(f'{exp_dir}/expression_val_index.json', 'w') as f:
                json.dump([int(s) for s in self.val_dataset.expression_steps], f)
            with open(f'{exp_dir}/subject_val_pids.json', 'w') as f:
                json.dump([int(s) for s in self.val_dataset.subject_IDs], f)
            with open(f'{exp_dir}/subject_val_inds.json', 'w') as f:
                json.dump([int(s) for s in self.val_dataset.subject_training_ordering], f)



        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        # Constructing Networks.
        # -------------------------------------------------------------------------------------------------------------
        # Set-up model itself
        id_decoder = prepped_entities['id_decoder']
        ex_decoder = prepped_entities['ex_decoder']

        # Initializing latent codes.
        self.latent_codes = prepped_entities['latent_codes'].to(device)
        self.latent_codes_val = prepped_entities['latent_codes_val'].to(device)
        id_decoder = id_decoder.to(device)
        ex_decoder = ex_decoder.to(device)
        self.id_decoder = id_decoder
        self.ex_decoder = ex_decoder


        if self.rank == 0:
            print('Number of Parameters in ID decoder: {}'.format(count_parameters(self.id_decoder)))
            print('Number of Parameters in EXPRESSION decoder: {}'.format(count_parameters(self.ex_decoder)))
        self.n3dmm = nn3dmm(
                        id_model=id_decoder,
                        ex_model=ex_decoder,
                        expr_direction='backward',
                        neutral_only=neutral_only
        )


        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        # Distribute Across GPUs.
        # -------------------------------------------------------------------------------------------------------------
        if self.rank == 0 and num_gpus > 1:
            print(f'Distributing across {num_gpus} GPUs...')
        ddp_modules = dict()
        for name, module in [('ID', self.id_decoder), ('EXP', self.ex_decoder),
                             ('lat_train', self.latent_codes), ('lat_val', self.latent_codes_val)]:
            if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            if name is not None and module is not None:
                ddp_modules[name] = module


        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        # Set-up Optimizers.
        # -------------------------------------------------------------------------------------------------------------
        paramter_list = []
        paramter_list += list(self.id_decoder.parameters())
        if not neutral_only:
            paramter_list += list(self.ex_decoder.parameters())
        self.optimizer_networks = optim.AdamW(params=paramter_list,
                                              lr=self.cfg['lr'],
                                              weight_decay=self.cfg['weight_decay'])
        lat_keys_id = ['geo', 'app']
        lat_keys_expr = ['exp']
        lat_params_id = []
        lat_params_expr = []
        for k in self.latent_codes.codebook.keys():
            if k in lat_keys_id:
                lat_params_id += list(self.latent_codes.codebook[k].parameters())
                if self.latent_codes.variational:
                    lat_params_id += list(self.latent_codes.codebook_logvar[k].parameters())

            if k in lat_keys_expr:
                lat_params_expr += list(self.latent_codes.codebook[k].parameters())
                if self.latent_codes.variational:
                    lat_params_expr += list(self.latent_codes.codebook_logvar[k].parameters())
        lat_params_id_val = []
        lat_params_expr_val = []
        for k in self.latent_codes_val.codebook.keys():
            if k in lat_keys_id:
                lat_params_id_val += list(self.latent_codes_val.codebook[k].parameters())
                if self.latent_codes_val.variational:
                    lat_params_id_val += list(self.latent_codes_val.codebook_logvar[k].parameters())
            if k in lat_keys_expr:
                lat_params_expr_val += list(self.latent_codes_val.codebook[k].parameters())
                if self.latent_codes_val.variational:
                    lat_params_expr_val += list(self.latent_codes_val.codebook_logvar[k].parameters())

        self.optimizer_lat = optim.SparseAdam(lat_params_id, lr=self.cfg['lr_lat'], )
        self.optimizer_lat_val = optim.SparseAdam(lat_params_id_val, lr=self.cfg['lr_lat'])
        if not neutral_only:
            self.optimizer_lat_expr = optim.SparseAdam(lat_params_expr, lr=self.cfg['lr_lat_expr'], )
            self.optimizer_lat_val_expr = optim.SparseAdam(lat_params_expr_val, lr=self.cfg['lr_lat_expr'])


        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']
        self.lr_lat_expr = self.cfg['lr_lat_expr']


        self.exp_path = exp_dir
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(cfg['exp_name'])
        if self.rank == 0:
            if not os.path.exists(self.checkpoint_path):
                print(self.checkpoint_path)
                os.makedirs(self.checkpoint_path)

        if num_gpus > 1:
            torch.distributed.barrier()

        self.val_min = None

        if self.rank == 0:
            config = self.log_dict(cfg)

        print('Big Box')
        self.min = [-0.4, -0.5, -0.7]
        self.max = [0.4, 0.7, 0.5]

        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).float()
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3))



        self.log_steps = 0
        self.step = 0

        self.disable_wandb = disable_wandb
        if self.rank == 0 and not self.disable_wandb:
            wandb.init(project=cfg['project_name'],
                       config=config,
                       name=cfg['exp_name'])
            wandb.watch(self.id_decoder, log_freq=1000)
            wandb.watch(self.ex_decoder, log_freq=1000)


    def reduce_lr(self, epoch):
        decay_steps = [350, 2*350, 3*350 ] + list(range(3*350+250, 250*100, 250))

        if self.neutral_only:
            decay_condition = epoch > 0 and self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0
        else:
            decay_condition = epoch in decay_steps
        if decay_condition:
            for param_group in self.optimizer_networks.param_groups:
                param_group["lr"] *= self.cfg['lr_decay_factor']
                print(f'decayed LR to {param_group["lr"]}')

        if decay_condition:
            for param_group in self.optimizer_lat.param_groups:
                param_group["lr"] *= self.cfg['lr_decay_factor']
                print(f'decayed LR to {param_group["lr"]}')

            for param_group in self.optimizer_lat_val.param_groups:
                param_group["lr"] *= self.cfg['lr_decay_factor']
                print(f'decayed LR to {param_group["lr"]}')

            if not self.neutral_only:
                for param_group in self.optimizer_lat_expr.param_groups:
                    param_group["lr"] *= self.cfg['lr_decay_factor']
                    print(f'decayed LR to {param_group["lr"]}')

                for param_group in self.optimizer_lat_val_expr.param_groups:
                    param_group["lr"] *= self.cfg['lr_decay_factor']
                    print(f'decayed LR to {param_group["lr"]}')


    def train_step(self, batch, epoch):

        self.id_decoder.train()
        self.ex_decoder.train()
        self.optimizer_networks.zero_grad()
        self.optimizer_lat.zero_grad()
        if not self.neutral_only:
            self.optimizer_lat_expr.zero_grad()


        loss_dict_nphm = compute_loss(batch,
                                          self.n3dmm,
                                          self.latent_codes,
                                          epoch,
                                          device=self.device,
                                            )


        loss_tot = 0
        for key in loss_dict_nphm.keys():
            loss_tot += self.cfg['lambdas'][key] * loss_dict_nphm[key]

        loss_tot.backward()

        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.id_decoder.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.ex_decoder.parameters(), max_norm=self.cfg['grad_clip'])

        self.optimizer_networks.step()
        self.optimizer_lat.step()
        if not self.neutral_only:
            self.optimizer_lat_expr.step()

        loss_dict = {k: loss_dict_nphm[k].item() for k in loss_dict_nphm.keys()}

        loss_dict.update({'loss': loss_tot.item()})

        return loss_dict


    def train_model(self, epochs):
        ckp_interval = self.cfg['ckpt_interval']

        for epoch in range(0, epochs):

            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss': 0.0})

            # optimze one epoch
            for batch in self.train_dataset_loader:
                loss_dict = self.train_step(batch, epoch)

                if self.rank == 0:
                    for k in loss_dict:
                        sum_loss_dict[k] += loss_dict[k]

            # periodically save checkpoint and reconstruct train/val examples
            if (epoch % ckp_interval == 0 and epoch > 0) or epoch == 1:
                if self.num_gpus > 1:
                    torch.distributed.barrier()
                self.save_checkpoint(epoch)
                self.log_recs(epoch, mode='train')
                if self.num_gpus > 1:
                    torch.distributed.barrier()
                self.log_recs(epoch, mode='val')
                if self.num_gpus > 1:
                    torch.distributed.barrier()


            if self.rank == 0:
                n_train = len(self.train_dataset_loader)
                for k in sum_loss_dict.keys():
                    sum_loss_dict[k] /= n_train

            # log best validation loss
            if len(self.val_dataset) > 1:
                val_loss_dict = self.val_step(epoch)

                if self.rank == 0:
                    if self.val_min is None:
                        self.val_min = val_loss_dict['loss']

                    if val_loss_dict['loss'] < self.val_min:
                        self.val_min = val_loss_dict['loss']
                        for path in glob(self.exp_path + 'val_min=*'):
                            os.remove(path)
                        np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss_dict['loss']])


            # print progress and log to WandB
            if self.rank == 0:
                print_str = "Epoch: {:5d}".format(epoch)
                for k in sum_loss_dict:
                    if len(self.val_dataset) > 1:
                        print_str += " " + k + " {:06.4f} - {:06.4f}".format(sum_loss_dict[k], val_loss_dict[k])
                    else:
                        print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
                print(print_str)

                if len(self.val_dataset) > 1:
                    sum_loss_dict.update({'val_' + k: v for (k,v) in zip(val_loss_dict.keys(), val_loss_dict.values())})

                if not self.disable_wandb:
                    wandb.log(sum_loss_dict, step=self.step)
                self.step += 1

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(epoch)


    def save_checkpoint(self, epoch):
        if self.rank == 0:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
            if not os.path.exists(path):
                state = {
                    'epoch': epoch,
                     'id_decoder': self.id_decoder.state_dict(),
                     'ex_decoder': self.ex_decoder.state_dict(),
                     'optimizer': self.optimizer_networks.state_dict(),
                     'latent_codes': self.latent_codes.state_dict(),
                     'latent_codes_val': self.latent_codes_val.state_dict(),
                     }
                if not self.neutral_only:
                    state['optimizer_lat_expr'] = self.optimizer_lat_expr.state_dict()
                    state['optimizer_lat_val_expr'] = self.optimizer_lat_val_expr.state_dict()
                state['optimizer_lat'] = self.optimizer_lat.state_dict(),
                state['optimizer_lat_val'] = self.optimizer_lat_val.state_dict(),


                torch.save(state, path)


    def val_step(self, epoch):
        self.id_decoder.eval()
        self.ex_decoder.eval()

        sum_val_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
        sum_val_loss_dict.update({'loss': 0.0})

        c = 0
        for val_batch in self.val_dataset_loader:
            self.optimizer_lat_val.zero_grad()
            if not self.neutral_only:
                self.optimizer_lat_val_expr.zero_grad()
            l_dict= compute_loss(val_batch, self.n3dmm, self.latent_codes_val, epoch, device=self.device)
            for k in l_dict.keys():
                sum_val_loss_dict[k] += l_dict[k].item()
            val_loss = 0.0
            for key in l_dict.keys():
                val_loss += self.cfg['lambdas'][key] * l_dict[key]
            val_loss.backward()
            self.optimizer_lat_val.step()
            if not self.neutral_only:
                self.optimizer_lat_val_expr.step()

            sum_val_loss_dict['loss'] += val_loss.item()
            c = c + 1

        for k in sum_val_loss_dict.keys():
            sum_val_loss_dict[k] /= c
        return sum_val_loss_dict


    def log_dict(self, cfg):
        return cfg

    def log_recs(self, epoch, mode='val'):
        self.id_decoder.eval()
        self.ex_decoder.eval()
        exp_dir = self.exp_path + 'recs/epoch_{}/'.format(epoch)
        os.makedirs(exp_dir, exist_ok=True)

        gt_imgs = []
        pred_imgs = []
        len_traindataset = len(self.train_dataset)
        if self.MIRROR:
            len_traindataset  = len_traindataset // 2
        num_steps = min(5, len_traindataset) # at most reconstruct 5 heads

        if mode == 'val':
            latent_codes = self.latent_codes_val
            dataset = self.val_dataset
        else:
            latent_codes = self.latent_codes
            dataset = self.train_dataset

        render_loader = []
        len_dataset = len(dataset)
        if self.MIRROR:
            len_dataset = len_dataset // 2
        perm = np.random.permutation(np.arange(len_dataset))
        for i in range(min(num_steps, len_dataset)):
            rndi = perm[i]
            render_loader.append({
                'iden_idx': torch.tensor(dataset.subject_training_ordering[rndi]),
                'expr': torch.tensor(dataset.expression_steps[rndi]),
                'iden': torch.tensor(dataset.subject_IDs[rndi]),
                'idx': rndi if not self.MIRROR else rndi*2,

            })
        for jj, render_batch in enumerate(render_loader):

            if jj > num_steps:
                break
            if self.rank == 0:
                print('log step', jj)

            try:
                iden_val = render_batch['iden_idx'].item()
                app_val = render_batch['iden_idx'].item()
                actual_iden = render_batch['iden'].item()
                expr_val = render_batch['expr'].item()
                step = render_batch['idx'].item()

                print('iden', iden_val)
                print('expr', expr_val)
                for mod in latent_codes.codebook.keys():
                    print(mod, latent_codes.codebook[mod].embedding.weight)

                encoding_val = latent_codes({'geo': torch.tensor([[iden_val]]).to(self.device),
                                             'app': torch.tensor([[app_val]]).to(self.device),
                                             'exp': torch.tensor([[step]]).to(self.device)})


                # render image
                if self.include_app:

                    iden = render_batch['iden'].item()
                    if self.include_app:
                        expr = render_batch['expr'].item()
                    else:
                        expr = None
                    val_gt_img, val_pred_img, trim_gt_val = self.render(encoding_val, iden % 10000, expr)

                    val_pred_img.save(exp_dir + 'img_{}_s{}_e_{}_pred.png'.format(mode, actual_iden, expr_val))
                    val_gt_img.save(exp_dir + 'img_{}_s{}_e_{}_gt.png'.format(mode, actual_iden, expr_val))
                    gt_imgs.append(np.array(val_gt_img))
                    pred_imgs.append(np.array(val_pred_img))

                torch.cuda.empty_cache()
                # reconstruct mesh
                logits_val = get_logits(decoder=self.n3dmm,
                                        encoding=encoding_val,
                                        grid_points=self.grid_points.clone(),
                                        nbatch_points=150000 if not os.path.exists('/mnt/rohan/') else 25000,
                                        )
                trim_val = mesh_from_logits(logits_val, self.min, self.max, self.res)

                # paint mesh with vertex colors
                if self.include_app and trim_val is not None and trim_val.vertices.shape[0] > 0:
                    vc_val = get_vertex_color(decoder=self.n3dmm,
                                              encoding=encoding_val,
                                              vertices=torch.from_numpy(trim_val.vertices).float(),
                                              nbatch_points=150000 if not os.path.exists('/mnt/rohan/') else 25000,
                                              uniform_scaling=True,
                                              device=self.device,
                                              )
                    trim_val.visual = trimesh.visual.ColorVisuals(trim_val, vertex_colors=vc_val)


                torch.cuda.empty_cache()
                # TODO proper naming
                if trim_val is not None:
                    trim_val.export(exp_dir + 'mesh_{}_s{}_e_{}.ply'.format(mode, actual_iden, expr_val))


            except Exception as e:
                print(traceback.format_exc())

        if self.num_gpus > 1:
            torch.distributed.barrier()

        # log images to wandb
        if self.include_app and self.rank == 0:
            for jj in range(len(gt_imgs)):
                gt_img = gt_imgs[jj]
                pred_img = pred_imgs[jj]
                if gt_img is not None:
                    cat_img = wandb.Image(np.concatenate([np.array(gt_img), np.array(pred_img)], axis=1))
                    wandb.log({'{}_renderings'.format(mode): cat_img,
                               }, step=self.step)

        self.log_steps += 1
        return


    # render image by rendering ground truth mesh using pyrender
    # then backproject points to 3D and query model to predict color for each backprojected pixel
    def render(self, encoding, iden, expr = None):

        m = self.manager.get_raw_mesh(subject=iden,
                                      expression=self.manager.get_neutral_expression(iden, neutral_type='closed') if expr is None else expr
                                      )
        valid = self.manager.cut_throat(m.vertices,
                                        subject=iden,
                                        expression=self.manager.get_neutral_expression(iden, neutral_type='closed')if expr is None else expr
                                        )

        m = cut_trimesh_vertex_mask(m, valid)

        rgb, points3d = render_and_backproject(m, down_scale_factor=2, crop=50)
        rgb = torch.from_numpy(rgb.copy())
        points3d = torch.from_numpy(points3d.copy())

        # predict image color for backprojected pixels
        pred_rgb = get_image_color(decoder=self.n3dmm,
                                   sample_points=points3d.float().view(-1, 3).unsqueeze(0),
                                   encoding=encoding,
                                   rend_size=[rgb.shape[1], rgb.shape[0]],
                                   uniform_scaling=True,
                                   device=self.device,
                                   )

        gt_rgb = Image.fromarray(rgb.detach().squeeze().cpu().numpy())
        return gt_rgb, pred_rgb, m