import torch
import torch.nn as nn
import numpy as np
from typing import Optional

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


class GlobalFieldNew(nn.Module):
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            nlayers=8,
            out_dim=1,
            num_freq_bands=None,
            freq_exp_base=0.5,
            input_dim=3,
            color_branch=False,
            n_hyper : int = 0,
            sdf_corrective : bool = False,
            pass_pos_to_app_head : bool  = False,
            lat_dim_app : int = 0,
            num_freq_bands_color=None,
            freq_exp_base_color=2.0,
            lat_dim_exp : int = 0,
            nlayers_color : int = 8,
            is_monolith : bool = False,
            communication_dim : int = 0,
            uv_communication : bool = False,
            include_anchors : bool = False,
        anchors = None,

    ):
        super().__init__()
        self.lat_dim = lat_dim
        self.lat_dim_app = lat_dim_app
        self.lat_dim_exp = lat_dim_exp
        self.input_dim = input_dim
        self.color_branch = color_branch
        self.pass_pos_to_app_head = pass_pos_to_app_head
        self.is_monolith = is_monolith
        out_dim += n_hyper
        if sdf_corrective:
            out_dim += 1
        self.n_hyper = n_hyper
        self.sdf_corrective = sdf_corrective
        self.num_freq_bands_geo = num_freq_bands
        self.freq_exp_base_geo = freq_exp_base
        self.num_freq_bands_color = num_freq_bands_color
        self.freq_exp_base_color = freq_exp_base_color

        self.communication_dim = communication_dim
        self.uv_communication = uv_communication
        if self.uv_communication:
            self.communication_dim = 0
            communication_dim = 0

        self.geo_mlp = DeepSDF(lat_dim=self.lat_dim,
                               hidden_dim=hidden_dim,
                               nlayers=nlayers,
                               geometric_init=True,
                               out_dim=1,
                               return_last_feats=True,
                               num_freq_bands=self.num_freq_bands_geo,
                               freq_exp_base=self.freq_exp_base_geo,
                               input_dim=self.input_dim,
                               )


        if self.color_branch:
            if self.communication_dim > 0:
                self.communcation_bottleneck1 = nn.Linear(in_features=hidden_dim,
                                                                out_features=16
                                                                          ).float()
                self.activation_bottleneck = torch.nn.ReLU()
                self.communcation_bottleneck2 = nn.Linear(in_features=16,
                                                                out_features=16
                                                                ).float()

                #if self.num_freq_bands_color > 0:
                #    fun = lambda x: self.freq_exp_base_color ** x
                #    self.freq_bands_color = fun(torch.arange(self.num_freq_bands_color))
                #    communication_dim = 16*(2*self.num_freq_bands_color+1)
                #d_in_color = communication_dim + self.lat_dim_app + self.lat_dim_exp
                if self.pass_pos_to_app_head:
                    communication_dim =  self.communication_dim + input_dim
            elif self.uv_communication:
                communication_dim = 2
                self.uv_mlp = DeepSDF(lat_dim=self.lat_dim,
                               hidden_dim=hidden_dim,
                               nlayers=6,
                               geometric_init=False,
                               out_dim=2,
                               return_last_feats=False,
                               input_dim=3
                               )

                self.uv_mlp_inverse = DeepSDF(lat_dim=self.lat_dim,
                               hidden_dim=hidden_dim,
                               nlayers=6,
                               geometric_init=False,
                               out_dim=3,
                               return_last_feats=False,
                               input_dim=2
                               )
            else:
                communication_dim = self.input_dim
            self.color_mlp = DeepSDF(lat_dim=self.lat_dim_app + self.lat_dim_exp,
                                   hidden_dim=hidden_dim,
                                   nlayers=nlayers_color,
                                   geometric_init=False,
                                   input_dim=communication_dim,
                                   out_dim=3,
                                   return_last_feats=False,
                                   num_freq_bands=self.num_freq_bands_color,
                                   freq_exp_base=self.freq_exp_base_color,
                                   beta=0,
                                   )


        self.include_anchors = include_anchors
        if self.include_anchors:
            self.mlp_pos = torch.nn.Sequential(
                torch.nn.Linear(self.lat_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 65 * 3)
            )
            self.anchors = anchors.squeeze(0)

    def forward(self, in_dict, anchors=None, skip_color=False):

        xyz = in_dict['queries_can']
        lat_rep = in_dict['cond']['geo']
        if self.is_monolith:
            lat_rep = torch.cat([lat_rep, in_dict['cond']['exp']], dim=-1)

        x, last_feats = self.geo_mlp(xyz, lat_rep)

        if self.color_branch and not skip_color:
            lat_rep_color = in_dict['cond']['app']

            if self.lat_dim_exp > 0:
                lat_rep_exp = in_dict['cond']['exp'][..., :self.lat_dim_exp]
                lat_rep_color = torch.cat([lat_rep_color, lat_rep_exp], dim=-1)

            if self.communication_dim > 0 and hasattr(self, 'communcation_bottleneck1'):
                color_comm = self.communcation_bottleneck1(last_feats)
                color_comm = self.activation_bottleneck(color_comm)
                color_comm = self.communcation_bottleneck2(color_comm)
                if self.pass_pos_to_app_head:
                    color_comm = torch.cat([color_comm, xyz], dim=-1)
            elif self.uv_communication:
                color_comm, _ = self.uv_mlp(xyz, lat_rep)
            else:
                color_comm = xyz
            color_preds, _ = self.color_mlp(color_comm, lat_rep_color)
            ret_dict = {'sdf': x, 'anchors': None, 'color': color_preds}
        else:
            ret_dict = {'sdf': x, 'anchors': None}

        if self.include_anchors:
            glob_feats = in_dict['cond']['geo'][:, 0, :]
            offsets = self.mlp_pos(glob_feats).view(-1, 65, 3)
            ret_dict['anchors'] = self.anchors.to(offsets.device) + offsets

        return ret_dict

    def get_reg_loss(self, latent_code, stds=None):
        loss_dict = {}
        loss_dict['reg_global_geo'] = (torch.norm(latent_code['geo'], dim=-1) ** 2).mean()
        loss_dict['reg_global_app'] = (torch.norm(latent_code['app'], dim=-1) ** 2).mean()
        return loss_dict

    def get_anchors(self, cond):
        if self.include_anchors:
            glob_feats = cond[:, 0, :]
            offsets = self.mlp_pos(glob_feats).view(-1, 65, 3)
            return self.anchors.to(offsets.device) + offsets
        else:
            return None

class GlobalField(nn.Module):
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
            freq_exp_base=0.5,
            input_dim=3,
            return_last_feats=False,
            color_branch=False,
            n_hyper : int = 0,
            sdf_corrective : bool = False,
            pass_pos_to_app_head : bool  = False,
            lat_dim_app : int = 0,
            num_freq_bands_color=10,
            freq_exp_base_color=2.0,
            lat_dim_exp : int = 0,
            is_monolith : bool = False,


    ):
        super().__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.lat_dim_app = lat_dim_app
        self.lat_dim_exp = lat_dim_exp
        self.input_dim = input_dim
        self.color_branch = color_branch
        self.pass_pos_to_app_head = pass_pos_to_app_head
        self.is_monolith = is_monolith
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
        self.freq_exp_base = freq_exp_base
        if color_branch:
            self.num_freq_bands_color = num_freq_bands_color
            self.freq_exp_base_color = freq_exp_base_color
        self.return_last_feats = return_last_feats
        if num_freq_bands is not None:
            fun = lambda x: freq_exp_base ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        for layer in range(0, self.num_layers - 1):

            #if layer + 1 in self.skip_in:
            #    out_dim = dims[layer + 1] - d_in
            #else:
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
            self.communcation_bottleneck1 = nn.Linear(in_features=hidden_dim,
                                                            out_features=16
                                                                      ).float()
            self.activation_bottleneck = torch.nn.ReLU()
            self.communcation_bottleneck2 = nn.Linear(in_features=16,
                                                            out_features=16
                                                            ).float()

            communication_dim = 16
            if self.num_freq_bands_color > 0:
                fun = lambda x: self.freq_exp_base_color ** x
                self.freq_bands_color = fun(torch.arange(self.num_freq_bands_color))
                communication_dim = 16*(2*self.num_freq_bands_color+1)
            d_in_color = communication_dim + self.lat_dim_app + self.lat_dim_exp
            if self.pass_pos_to_app_head:
                d_in_color += d_in_spatial
            self.color_mlp = nn.Sequential(
                nn.Linear(d_in_color, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )

    def forward(self, in_dict, anchors=None):

        xyz = in_dict['queries_can']
        lat_rep = in_dict['cond']['geo']
        if hasattr(self, 'is_monolith') and self.is_monolith:
            lat_rep = torch.cat([lat_rep, in_dict['cond']['exp']], dim=-1)
        lat_rep = lat_rep[:, :1, :]
        if lat_rep.shape[1] == 1 and xyz.shape[1] > 1:
            lat_rep = lat_rep.repeat(1, xyz.shape[1], 1)
        if self.color_branch:
            lat_rep_color = in_dict['cond']['app']

            lat_rep_color = lat_rep_color[:, :1, :]
            if self.lat_dim_exp > 0:
                lat_rep_exp = in_dict['cond']['exp'][:, :1, :]
                lat_rep_color = torch.cat([lat_rep_color, lat_rep_exp], dim=-1)
            if lat_rep_color.shape[1] == 1 and xyz.shape[1] > 1:
                lat_rep_color = lat_rep_color.repeat(1, xyz.shape[1], 1)

        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        batch_size, num_points, dim_in = xyz.shape

        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
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
            if self.pass_pos_to_app_head:
                if self.num_freq_bands is not None:
                    color_cond = torch.cat([pos_embed, last_feats, lat_rep_color], dim=-1)
                else:
                    color_cond = torch.cat([xyz, last_feats, lat_rep_color], dim=-1)
            else:
                color_comm = self.communcation_bottleneck1(last_feats)
                color_comm = self.activation_bottleneck(color_comm)
                color_comm = self.communcation_bottleneck2(color_comm)
                comm_embeds = [color_comm]
                for freq in self.freq_bands_color:
                    comm_embeds.append(torch.sin(color_comm * freq))
                    comm_embeds.append(torch.cos(color_comm * freq))

                comm_embed = torch.cat(comm_embeds, dim=-1)
                color_cond = torch.cat([comm_embed, lat_rep_color], dim=-1)

            color_preds = self.color_mlp(color_cond)

            return {'sdf': x, 'anchors': None, 'color': color_preds}

        return {'sdf': x, 'anchors': None}


def sample_point_feature(q, p, fea, var=0.1**2, background=False):
    # q: B x M x 3
    # p: B x N x 3
    # fea: B x N x c_dim
    # p, fea = c

    #print(q.shape)
    #print(p.shape)
    #print(fea.shape)
    # distance betweeen each query point to the point cloud
    dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
    if background:
        dist_const = torch.ones_like(dist[:, :, :1])* (-0.2)#(-0.025) #hair 0.2
        dist = torch.cat([dist, dist_const], dim=-1)

    weight = (dist / var).exp()  # Guassian kernel

    # weight normalization
    weight = weight / (weight.sum(dim=2).unsqueeze(-1) + 1e-6)
    #print(weight.shape)
    #print(fea.shape)
    #c_out = weight @ fea  # B x M x c_dim
    c_out = (weight.unsqueeze(-1) * fea).sum(dim=2)
    return c_out


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

        if self.local_arch:
            raise ValueError('Not Supported: ensemble of MLPs for deformation network')
        else:
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

            if not self.local_arch:
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

