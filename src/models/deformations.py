import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from mononphm.models.canonical_space import NPHM


class DeepSDF(nn.Module):
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            nlayers=8,
            geometric_init=True,
            radius_init=1,
            beta=100,
            out_dim=1,
            num_freq_bands=None,
            input_dim=3,
            return_last_feats=False,
            color_branch=False,
            n_hyper : int = 0,
            sdf_corrective : bool = False,
            freq_exp_base : float = 2.0,
            legacy : bool = False,

    ):
        super().__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        self.color_branch = color_branch
        out_dim += n_hyper
        if sdf_corrective:
            out_dim += 1
        self.n_hyper = n_hyper
        self.sdf_corrective = sdf_corrective
        print(f'Creating DeepSDF with input dim f{d_in}, hidden_dim f{hidden_dim} and output_dim {out_dim}')

        dims = [hidden_dim] * nlayers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2]
        self.num_freq_bands = num_freq_bands
        self.return_last_feats = return_last_feats
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        for layer in range(0, self.num_layers - 1):

            if legacy:
                if layer + 1 in self.skip_in:
                    out_dim = dims[layer+1] - d_in
                else:
                    out_dim = dims[layer+1]

                in_dim = dims[layer]
            else:
                if layer in self.skip_in:
                    in_dim = dims[layer] + d_in
                else:
                    in_dim = dims[layer]

                out_dim = dims[layer + 1]

            lin = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                #else:
                #    torch.nn.init.constant_(lin.bias, 0.0)

                #    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #else:
            #    stdv = 1. / math.sqrt(lin.weight.size(1))
            #    #stdv = stdv / 5
            #    print('Attention: using lower std to init Linear layer!!')
            #    lin.weight.data.uniform_(-stdv, stdv)
            #    if lin.bias is not None:
            #        lin.bias.data.uniform_(-stdv, stdv)
            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

        if self.color_branch:
            self.return_last_feats = True
            self.color_mlp = nn.Sequential(
                nn.Linear(hidden_dim + d_in_spatial, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )

    def forward(self, xyz, lat_rep, anchors=None):
        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz * freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            if lat_rep.shape[1] == 1 and pos_embed.shape[1] > 1:
                lat_rep = lat_rep.repeat(1, pos_embed.shape[1], 1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
            if lat_rep.shape[1] == 1 and xyz.shape[1] > 1:
                lat_rep = lat_rep.repeat(1, xyz.shape[1], 1)
            inp = torch.cat([xyz, lat_rep], dim=-1)
        x = inp
        last_feats = None

        for layer in range(0, self.num_layers - 1):
            #print(x.shape)
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)
            if self.return_last_feats and layer == self.num_layers - 3:
                last_feats = x

        if self.color_branch:
            if self.num_freq_bands is not None:
                color_cond = torch.cat([pos_embed, last_feats], dim=-1)
            else:
                color_cond = torch.cat([xyz, last_feats], dim=-1)
            color_preds = self.color_mlp(color_cond)

            return x, None, color_preds



        return x, last_feats



class DeformationNetwork(nn.Module):
    def __init__(
            self,
            mode,
            lat_dim_expr,
            lat_dim_id,
            lat_dim_glob_shape,
            lat_dim_loc_shape,
            n_loc,
            anchors,
            hidden_dim,
            nlayers=8,
            out_dim=3,
            input_dim=3,
            neutral_only=False,
            sdf_corrective : bool = False,
            n_hyper : int = 0,
            local_arch : bool = False,
            legacy : bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.local_arch = local_arch
        self.lat_dim_glob_shape = lat_dim_glob_shape
        self.lat_dim_loc_shape = lat_dim_loc_shape
        self.lat_dim_id = lat_dim_id

        self.neutral_only = neutral_only
        self.sdf_corrective = sdf_corrective
        self.n_hyper = n_hyper

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim
        self.num_kps = n_loc
        self.out_dim = out_dim + self.n_hyper
        if self.sdf_corrective:
            self.out_dim += 1


        self.lat_dim_expr = lat_dim_expr


        if not self.neutral_only and not self.local_arch:

            if self.mode == 'glob_only':
                self.lat_dim = lat_dim_glob_shape + lat_dim_expr
            elif self.mode == 'expr_only':
                self.lat_dim = lat_dim_expr
            elif self.mode == 'interpolate':
                self.lat_dim = lat_dim_glob_shape + lat_dim_expr + lat_dim_loc_shape
            elif self.mode == 'compress':
                self.lat_dim = lat_dim_expr + lat_dim_id
                self.compressor = nn.Sequential(
                nn.Linear((lat_dim_loc_shape + 3) * (n_loc) + lat_dim_loc_shape + lat_dim_glob_shape, lat_dim_id))

            elif self.mode == 'GNN':
                self.lat_dim = lat_dim_expr * 2

                self.pos_enc = nn.Sequential(nn.Linear(3, lat_dim_loc_shape), nn.ReLU(), nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape))
                self.local_combiner = nn.Sequential(nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape), nn.ReLU(), nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape))
                self.global_combiner = nn.Sequential(nn.Linear(self.lat_dim_glob_shape + (n_loc)*lat_dim_loc_shape, 512),
                                                     nn.ReLU(),
                                                     nn.Linear(512, lat_dim_expr))

            else:
                raise ValueError('Unknown mode!')

            print('creating DeepSDF with...')
            print('lat dim', self.lat_dim)
            print('hidden_dim', hidden_dim)
            self.defDeepSDF = DeepSDF(lat_dim=self.lat_dim,
                                      hidden_dim=hidden_dim,
                                      nlayers=nlayers,
                                      geometric_init=False,
                                      out_dim=self.out_dim,
                                      input_dim=input_dim,
                                      legacy=legacy).float()

        self.anchors = anchors


    def forward(self, in_dict):
        #xyz: B x N x 3
        #lat: B x N x lat_dim
        #anchors: B x N x n_kps x 3

        if self.neutral_only:
            return {'offsets': torch.zeros_like(in_dict['queries'])}

        xyz = in_dict['queries']
        lat_rep_id = in_dict['cond']['geo']
        lat_rep_ex = in_dict['cond']['exp']



        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        B, N, _ = xyz.shape

        if self.local_arch:
            # concatenate expr and id codes
            #cond = torch.cat([lat_rep_id, lat_rep_ex], dim=-1)
            cond = lat_rep_ex
            # fake input s.t. AnchoredEnsembleField can handle it
            in_dict_fake = {'queries_can': xyz, 'cond': {'geo': cond}}
            out_dict_fake = self.defDeepSDF(in_dict_fake)
            pred = out_dict_fake['sdf']
        else:
            if lat_rep_id.shape[1] == 1:
                lat_rep_id = lat_rep_id.repeat(1, N, 1)
            if lat_rep_ex.shape[1] == 1:
                lat_rep_ex = lat_rep_ex.repeat(1, N, 1)

            if self.mode == 'glob_only':
                cond = torch.cat([lat_rep_id[:, :, :self.lat_dim_glob_shape], lat_rep_ex], dim=-1)

            elif self.mode == 'expr_only':
                cond = lat_rep_ex
            elif self.mode == 'interpolate':
                assert 1 == 2
            elif self.mode == 'compress':
                anchors = in_dict['anchors']
                if not anchors.shape[1] == N:
                    if len(anchors.shape) != 4:
                        anchors = anchors.unsqueeze(1).repeat(1, N, 1, 1)
                    else:
                        anchors = anchors[:, 0, :, :].unsqueeze(1).repeat(1, N, 1, 1)
                elif N == anchors.shape[-2] and len(anchors.shape) != 4:
                    anchors = anchors.unsqueeze(1).repeat(1, N, 1, 1)
                concat = torch.cat([lat_rep_id, anchors.reshape(B, N, -1)], dim=-1)
                compressed = self.compressor(concat[:, 0, :]).unsqueeze(1).repeat(1, N, 1)
                #if self.training:
                #    compressed += torch.randn(compressed.shape, device=compressed.device) / 200

                cond = torch.cat([compressed, lat_rep_ex], dim=-1)


            elif self.mode == 'GNN':
                assert 1 == 2
            else:
                raise ValueError('Unknown mode')

            pred = self.defDeepSDF(xyz, cond)[0]

        output = {'offsets': pred[..., :3]}

        if self.sdf_corrective:
            output.update({'sdf_corrective': pred[..., 3+self.n_hyper:3+self.n_hyper+1]})
        if self.n_hyper > 0:
            output.update({'hyper_coords': pred[..., 3:3+self.n_hyper]})
        return output


class ZeroDeformation(nn.Module):
    '''
    Imitates the signature of DeformationNetwork, but always predicts zero offsets.
    '''
    def __init__(
            self,
    ):
        super().__init__()

        self.sdf_corrective = False
        self.n_hyper = 0
        self.lat_dim_expr = 5
        self.neutral_only = True

    def forward(self, in_dict):
        #xyz: B x N x 3
        #lat: B x N x lat_dim
        #anchors: B x N x n_kps x 3

        return {'offsets': torch.zeros_like(in_dict['queries'])}



def get_ex_model(cfg_ex : dict,
                 anchors
                 ):
    '''
    Instantiate a DeformationNetwork
    '''
    ex_model = DeformationNetwork(
        mode=cfg_ex['decoder']['ex']['mode'],
        lat_dim_expr=cfg_ex['decoder']['ex']['lat_dim_ex'],
        lat_dim_id=cfg_ex['decoder']['ex']['lat_dim_id'],
        lat_dim_glob_shape=cfg_ex['decoder']['id']['lat_dim_glob'],
        lat_dim_loc_shape=cfg_ex['decoder']['id']['lat_dim_loc_geo'],
        n_loc=cfg_ex['decoder']['id']['nloc'],
        anchors=anchors,
        hidden_dim=cfg_ex['decoder']['ex']['hidden_dim'],
        nlayers=cfg_ex['decoder']['ex']['nlayers'],
        out_dim=3,
        input_dim=3,
        neutral_only=False,
        n_hyper=cfg_ex['decoder']['n_hyper'],
        sdf_corrective=False,
        local_arch=False,
    )
    return ex_model