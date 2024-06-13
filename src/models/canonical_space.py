from typing import Tuple
import torch.nn

from torch_geometric.utils import index_sort
import numpy as np

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse
)

from pytorch3d.ops import knn_points

from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from mononphm import env_paths


def softmax_mod_varySize2(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    num_neighbors: int = 8,
    dim: int = 0,
    blend_std : float = 4.0,
) -> Tensor:
    '''
    In the neighborhood of each query point, compute the influence for each neighboring local MLP.
    Idea: Use a Gaussain Kernel, and set its standard deviation such that its influence becomes almost negligable once its furthest neighbour is reached.
    '''

    N = maybe_num_nodes(index, num_nodes)
    dists = src.view(-1, num_neighbors + 1)[:, :8] #num_neighbors] # exclude extra "non-local" MLP
    local_std = torch.max(dists, dim=-1, keepdim=False)[0] / blend_std
    local_std = torch.repeat_interleave(local_std, num_neighbors + 1)
    out =(-(src/local_std).square()/2).exp()

    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-8
    out_sum = out_sum.index_select(dim, index)

    return out / out_sum


class MLPDeepSDF(torch.nn.Module):
    def __init__(self,  d_in,
                        lat_dim: int,
                        hidden_dim: int,
                        n_layers: int=8,
                        beta: int=100,
                        out_dim: int=1,
                        return_color_feats: bool=False,
                        append_skip : bool = False,
                        without_skip : bool = False,
                    pass_every_layer : bool = False,
                 ):
        super().__init__()


        self.lat_dim = lat_dim
        self.return_color_feats = return_color_feats

        dims = [hidden_dim] * n_layers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [n_layers // 2]
        if without_skip:
            self.skip_in = []

        self.layers = torch.nn.ModuleList()

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                # out_dim = dims[layer + 1] #- d_in
                # in_dim = dims[layer] #+ d_in
                if append_skip:
                    out_dim = dims[layer + 1]
                else:
                    out_dim = dims[layer + 1] - d_in

                in_dim = dims[layer]
            elif layer in self.skip_in:
                if append_skip:
                    in_dim = dims[layer] + d_in
                else:
                    in_dim = dims[layer]
                out_dim = dims[layer + 1]
            else:
                if pass_every_layer and layer > 0 and layer < self.num_layers - 2:
                    out_dim = dims[layer + 1] #+ d_in - 3
                    in_dim = dims[layer] + d_in-3
                else:
                    out_dim = dims[layer + 1]
                    in_dim = dims[layer]

            if layer == self.num_layers - 2:
                activation_function = None
            else:
                if beta > 0:
                   activation_function = torch.nn.Softplus(beta=beta)
                else:
                   activation_function = torch.nn.ReLU()

            lin = torch.nn.Linear(in_features=in_dim,
                                      out_features=out_dim,
                                      )

            self.layers.append(lin)
            if activation_function is not None:
                self.layers.append(activation_function)


        self.reset_parameters()
        self.pass_every_layer = pass_every_layer


    def forward(self, x: Tensor, ) -> (Tensor, Tensor):


        inp = x
        color_feats = None
        for i, layer in enumerate(self.layers):

            if i in self.skip_in:
                x = torch.cat([x, inp], -1) / 1.414
            if self.pass_every_layer and i > 0 and i < len(self.layers) - 2 and i % 2 == 1:
                x = torch.cat([x, inp[..., :-3]], -1) / 1.414

            x = layer(x)

            if self.return_color_feats > 0 and i == len(self.layers) - 3:
                color_feats = x

        return x, color_feats



    def reset_parameters(self):
        for lin in self.layers:
            reset_fun = getattr(lin, 'reset_parameters', None)
            if callable(reset_fun):
                lin.reset_parameters()



class EnsembleDeepSDF(torch.nn.Module):
    def __init__(self,  ensemble_size: int,
                        n_symm: int,
                        lat_dim: int,
                        hidden_dim: int,
                        n_layers: int=8,
                        beta: int=100,
                        out_dim: int=1,
                        num_freq_bands: Optional[int]=None,
                        input_dim: int=3,
                        return_color_feats: bool=False,
                        disable_hetero_linear: bool=False,
                        sort_hetero: bool = False,
                        append_skip : bool = False,
                        without_skip : bool = False,
                        pass_every_layer : bool = False,
                 ):
        super().__init__()

        if num_freq_bands is None:
            d_in = input_dim + lat_dim
        else:
            d_in = input_dim * (2 * num_freq_bands + 1) + lat_dim
        self.ensemble_size = ensemble_size
        self.n_symm = n_symm
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        self.return_color_feats = return_color_feats

        dims = [hidden_dim] * n_layers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [n_layers // 2]
        if without_skip:
            self.skip_in = []
        self.num_freq_bands = num_freq_bands
        self.num_types = self.ensemble_size - self.n_symm + 1
        if disable_hetero_linear:
            self.num_types = 1
        #self.return_last_feats = return_last_feats
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.register_buffer('freq_bands', fun(torch.arange(num_freq_bands)))
            #self.freq_bands =  fun(torch.arange(num_freq_bands))

        self.layers = torch.nn.ModuleList()

        for layer in range(0, self.num_types):
            self.layers.append(MLPDeepSDF(d_in, self.lat_dim, hidden_dim, n_layers, beta, out_dim, return_color_feats, append_skip, without_skip, pass_every_layer=pass_every_layer))

        #if self.return_color_feats:
        #    self.color_feats_extractor = HeteroLinear(in_channels=hidden_dim,
        #                                             out_channels=self.return_color_feats,
        #                                              num_types=self.num_types,
        #                                              is_sorted=True,
        #                                              bias_initializer='uniform')



        #if beta > 0:
        #    self.activation = torch.nn.Softplus(beta=beta)

        # vanilla relu
        #else:
        #    self.activation = torch.nn.ReLU()

        self.reset_parameters()


    def forward(self, x: Tensor, type_vec: Tensor) -> (Tensor, Tensor):

        if self.num_freq_bands is not None:
            xyz = x[..., :self.input_dim]
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            x = torch.cat([pos_embed, x[..., self.input_dim:]], dim=-1)

        inp = x
        color_feats = None
        #print('layer', len(self.layers))
        #separating_indices = torch.where(type_vec[1:] - type_vec[:-1])[0]
        separating_indices = []
        for ii in range(self.num_types):

            active = (type_vec == ii)
            if active.sum() <= 0:
                separating_indices.append(None)
            else:
                separating_indices.append(active.nonzero(as_tuple=True)[0].max())
        return_list = []
        return_list_color_feats = []
        last_index = 0
        for i in range(len(self.layers)):
            if separating_indices[i] is None:
                continue
            if i == 0:
                _input = x[last_index:separating_indices[i].item()+1]
            #elif i == len(self.layers) - 1:
            #    _input = x[separating_indices[-2]:]
            else:
                _input = x[last_index:separating_indices[i].item()+1]
            last_index = separating_indices[i].item()+1
            member_out = self.layers[i](_input)
            return_list.append(member_out[0])
            return_list_color_feats.append(member_out[1])


        return torch.cat(return_list, dim=0), torch.cat(return_list_color_feats, dim=0) if self.return_color_feats else None



    def reset_parameters(self):
        for lin in self.layers:
            lin.reset_parameters()



class NPHM(MessagePassing):
    def __init__(self,
                 lat_dim_glob: int,
                 lat_dim_loc: int,

                 n_symm: int,
                 n_anchors: int,
                 anchors: torch.Tensor,

                 hidden_dim_geo: int,
                 hidden_dim_app: int,
                 n_layers_geo: int,
                 n_layers_app: int,
                 d_pos: int = 3,
                 pos_mlp_dim: int = 128,
                 num_neighbors: int = 8,
                 color_branch: bool = False,
                 num_freq_bands: Optional[int] = None,
                 color_communication : bool = True,
                 rank=0,
                 global_lat_color : bool = True,
                 is_monolith : bool = False,
                 blend_std : float = 4.0,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if rank is None:
            rank = 0
        self.anchors = anchors.squeeze(0)

        self.color_branch = color_branch
        self.color_communication = color_communication
        self.global_lat_color = global_lat_color
        self.is_monolith = is_monolith
        self.blend_std = blend_std


        self.lat_dim_glob = lat_dim_glob
        self.lat_dim_glob_geo = self.lat_dim_glob
        self.lat_dim_glob_app = self.lat_dim_glob
        self.lat_dim_loc = lat_dim_loc


        if not self.color_branch:
            self.lat_dim_loc_modality = lat_dim_loc
        else:
            self.lat_dim_loc_modality = lat_dim_loc // 2
            self.lat_dim_geo = self.lat_dim_loc_modality
            self.lat_dim_app = self.lat_dim_loc_modality

        self.lat_dim_loc_geo =self.lat_dim_loc_modality
        self.lat_dim_loc_app =self.lat_dim_loc_modality

        self.n_symm = n_symm
        self.num_symm_pairs = self.n_symm #alias
        self.num_kps = n_anchors #alias
        self.hidden_dim_geo = hidden_dim_geo
        self.hidden_dim_app = hidden_dim_app
        self.n_layers_geo = n_layers_geo
        self.n_layers_app = n_layers_app
        self.n_anchors = n_anchors
        self.pos_mlp_dim = pos_mlp_dim
        self.num_neighbors = num_neighbors
        self.lat_dim = lat_dim_glob + (n_anchors+1) * lat_dim_loc
        self.num_freq_bands = num_freq_bands if num_freq_bands is not None else 0

        self.num_types = self.num_kps - self.n_symm + 1

        #if self.color_communication:
        d_in_geo = self.lat_dim_loc_modality+self.lat_dim_glob
        if self.is_monolith:
            d_in_geo = 2*(self.lat_dim_loc_modality + self.lat_dim_glob)
        self.deepSDFensemble = EnsembleDeepSDF(ensemble_size=self.n_anchors,
                                               n_symm=self.n_symm,
                                               lat_dim=d_in_geo,
                                               hidden_dim=self.hidden_dim_geo,
                                               n_layers=self.n_layers_geo,
                                               return_color_feats=self.color_branch,
                                               input_dim=d_pos,
                                               without_skip=True,
                                               append_skip=True,
                                               pass_every_layer=False,  #True,
                                               ).float()
        if self.color_branch:
            if self.num_freq_bands > 0:
                fun = lambda x: 2 ** x
                self.freq_bands = fun(torch.arange(num_freq_bands))

            self.color_pass_pos = False
            if self.color_communication:
                self.communication_dim = 16

                self.communcation_bottleneck1 = torch.nn.Linear(
                                                            in_features=self.hidden_dim_geo,
                                                            out_features=self.communication_dim).float()
                self.activation_bottleneck = torch.nn.ReLU()
                self.communcation_bottleneck2 = torch.nn.Linear(
                                                            in_features=self.communication_dim,
                                                            out_features=self.communication_dim).float()
                dim_in_color_mlp = self.communication_dim + self.lat_dim_loc_modality
            else:
                # no comm. separated dim_in_color_mlp = self.lat_dim_loc_modality
                dim_in_color_mlp = 2*self.lat_dim_loc_modality + 3 + 3 * 2 * self.num_freq_bands
                assert self.num_freq_bands is None or self.num_freq_bands == 0
                self.communication_dim = d_pos # color network gets spatial coordinates as input



            app_out_dim = 3

            #self.app_ensemble = EnsembleDeepSDF(ensemble_size=self.n_anchors,
            self.aggr_app = EnsembleDeepSDF(ensemble_size=self.n_anchors,
                                                n_symm=self.n_symm,
                                                lat_dim=self.lat_dim_loc_modality if not self.global_lat_color else self.lat_dim_loc_modality + self.lat_dim_glob,
                                                hidden_dim=self.hidden_dim_app,
                                                n_layers=self.n_layers_app,
                                                return_color_feats=False,
                                                input_dim=self.communication_dim,
                                                out_dim=app_out_dim,
                                                beta=0,
                                                without_skip=True,
                                                append_skip=True,
                                                pass_every_layer=False,  #True,
                                                ).float()



        self.reset_parameters()
        self.is_symm = torch.ones([self.n_symm*2], dtype=torch.bool)
        self.is_symm[::2] = 0
        self.is_symm = torch.cat([self.is_symm, torch.zeros(self.n_anchors - 2*self.n_symm + 1, dtype=torch.bool, device=self.is_symm.device)])
        self.is_symm = self.is_symm.unsqueeze(-1).unsqueeze(0).to(rank)
        self.anchor_idx = torch.repeat_interleave(torch.arange(self.n_symm), 2)
        self.anchor_idx = torch.cat([self.anchor_idx, torch.arange(self.n_symm, self.n_anchors - self.n_symm + 1)])
        self.anchor_idx = self.anchor_idx.unsqueeze(-1).unsqueeze(0).to(rank)

        self.mlp_pos = torch.nn.Sequential(
            torch.nn.Linear(self.lat_dim_glob, self.pos_mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pos_mlp_dim, self.pos_mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pos_mlp_dim, self.n_anchors * 3)
        )

        propagate_type = {'feats': Tuple[Tensor, Tensor], 'color_feats': Tuple[Tensor, Tensor],
                          'glob_feats': Tuple[Tensor, Tensor], 'anchor_idx': Tensor, 'is_symm': Tensor, 'pos': Tuple[Tensor, Tensor],
                          'debug_region': bool}

    def reset_parameters(self):
        super().reset_parameters()

        for lin in self.deepSDFensemble.layers:
            lin.reset_parameters()

        if self.color_branch:
            #for lin in self.app_ensemble.layers:
            self.aggr_app.reset_parameters()

        #if self.color_branch:
        #    for lin in self.ensembled_color_mlp:
        #        lin.reset_parameters()


    '''
        def forward(self, xyz, lat_rep, anchors_gt):
        #xyz: B x N x 3
        #lat: B x N x 256
    '''

    def forward(self, in_dict, debug_regio=False, squeeze=False, skip_color=False):

        query_pos = in_dict['queries_can']

        # construct interleved latent code
        if self.color_branch:
            lat_rep = torch.stack([in_dict['cond']['geo'][..., self.lat_dim_glob:],
                                   in_dict['cond']['app'][..., self.lat_dim_glob:]], dim=-1)
            lat_rep = torch.reshape(lat_rep, (lat_rep.shape[0], lat_rep.shape[1], -1))
            lat_rep = torch.cat([in_dict['cond']['geo'][..., :self.lat_dim_glob], lat_rep], dim=-1)

        else:
            lat_rep = in_dict['cond']['geo']

        if self.is_monolith:
            lat_rep_exp = in_dict['cond']['exp']




        if len(query_pos.shape) < 3:
            query_pos = query_pos.unsqueeze(0)

        batch_size, num_points, dim_in = query_pos.shape

        # extract global and local features from lat_rep:
        glob_feats = lat_rep[:, 0, :self.lat_dim_glob]

        if self.color_branch:
            anchor_feats = lat_rep[:, 0, self.lat_dim_glob::2].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
            color_feats = lat_rep[:, 0, self.lat_dim_glob+1::2].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
        else:
            anchor_feats = lat_rep[:, 0, self.lat_dim_glob:].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
            color_feats = None

        if self.is_monolith:
            anchor_feats_exp = lat_rep_exp[:, 0, self.lat_dim_glob:].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
            anchor_feats = torch.cat([anchor_feats, anchor_feats_exp], dim=-1)

        anchor_pos = self.get_anchors(lat_rep)

        if self.is_monolith:
            glob_feats_exp = lat_rep_exp[:, 0, :self.lat_dim_glob]
            glob_feats = torch.cat([glob_feats, glob_feats_exp], dim=-1)
            glob_feats = (glob_feats.unsqueeze(1).repeat(1, self.n_anchors+1, 1).view(-1, self.lat_dim_glob*2), None)
        else:
            glob_feats = (glob_feats.unsqueeze(1).repeat(1, self.n_anchors+1, 1).view(-1, self.lat_dim_glob), None)
        if self.global_lat_color and self.color_branch:
            glob_feats_color = (in_dict['cond']['app'][:, 0, :self.lat_dim_glob].repeat(1, self.n_anchors+1, 1).view(-1, self.lat_dim_glob), None)
        else:
            glob_feats_color = None


        anchor_feats = (anchor_feats,  None)
        color_feats = (color_feats, None)


        # compute bi-partite knn graph
        knn_results = knn_points(query_pos[..., :3], anchor_pos, K=self.num_neighbors)

        knn_idx = torch.cat([knn_results.idx, (self.n_anchors) * torch.ones_like(knn_results.idx[:, :, :1])], dim=-1)

        anchor_idx_correction = torch.arange(batch_size, device=query_pos.device).unsqueeze(1).unsqueeze(1) * (self.n_anchors + 1)
        point_idx_correction = torch.arange(batch_size, device=query_pos.device).unsqueeze(1) * (num_points)
        knn_edges = torch.stack([(knn_idx + anchor_idx_correction).view(batch_size, -1),
                                 torch.repeat_interleave(torch.arange(num_points, device=query_pos.device),
                                                         self.num_neighbors+1, dim=0).unsqueeze(0).repeat(batch_size, 1) + point_idx_correction], dim=1)

        if debug_regio:
            self.latest_edges = knn_edges

        if query_pos.shape[-1] > 3:
            anchor_pos = torch.cat([anchor_pos, torch.zeros([anchor_pos.shape[0], anchor_pos.shape[1], query_pos.shape[-1]-3], dtype=anchor_pos.dtype, device=anchor_pos.device)], dim=-1)
        pos = (torch.cat([anchor_pos, torch.zeros_like(anchor_pos[:, :1, :])], dim=1), query_pos)

        # merge batch and point-sample dimensions
        edge_index = knn_edges.permute(1, 0, 2).reshape(2, -1)

        pos = (pos[0].view(-1, dim_in), pos[1].view(-1, dim_in))
        anchor_idx = (self.anchor_idx.repeat(batch_size, 1, 1).view(-1, 1), None)
        is_symm = (self.is_symm.repeat(batch_size, 1, 1).view(-1, 1), None)
        # run network

        propagate_type = {'feats': Tuple[Tensor, Tensor], 'color_feats': Tuple[Tensor, Tensor],
                          'glob_feats': Tuple[Tensor, Tensor], 'anchor_idx': Tensor, 'is_symm': Tensor,
                          'glob_feats': Tuple[Tensor, Tensor], 'anchor_idx': Tensor, 'is_symm': Tensor,
                          'pos': Tuple[Tensor, Tensor],
                          'debug_region': bool}
        # propagate_type: (feats: Tuple[Tensor, Tensor], color_feats: Tuple[Tensor, Tensor], glob_feats: Tuple[Tensor, Tensor], anchor_idx: Tensor, is_symm: Tensor, pos: Tuple[Tensor, Tensor], debug_region: bool)
        out = self.propagate(edge_index,
                             feats=anchor_feats,
                             color_feats=color_feats,
                             glob_feats=glob_feats,
                             glob_feats_color=glob_feats_color,
                             anchor_idx=anchor_idx,
                             is_symm=is_symm,
                             pos=pos,
                             debug_region=debug_regio,
                             size=None,
                             skip_color=skip_color,)


        if squeeze:
            out_dim = 4
            out = out.view(batch_size, num_points, 4)
        else:
            out = out.view(batch_size, num_points, out.shape[-1])

        if self.color_branch:

            out_color = out[..., 1:]

            out = out[..., :1]

            return {'sdf': out, 'anchors': anchor_pos[..., :3], 'color': out_color}
        else:
            return {'sdf': out, 'anchors': anchor_pos[..., :3]}



    def message(self,
                feats_j: Tensor,
                color_feats_j,
                glob_feats_j: Tensor,
                glob_feats_color_j: Tensor,
                anchor_idx_j: Tensor,
                is_symm_j: Tensor,
                pos_i: Tensor, pos_j: Tensor,
                index: Tensor,
                ptr: OptTensor,
                debug_region: bool=False,
                skip_color=False):


        delta = pos_i - pos_j

        # mirror symmetric points
        is_symmetric_idx = is_symm_j# (anchor_idx_j < self.n_symm*2) & (anchor_idx_j % 2 == 1)
        delta[is_symmetric_idx.squeeze(), 0] *= -1


        net_in = torch.cat([glob_feats_j, feats_j, delta], dim=-1)

        # sort anchor index here once due to nature of hetero linear implementation
        anchor_idx_j = anchor_idx_j.squeeze()
        anchor_idx_j_sorted, perm = index_sort(anchor_idx_j, self.deepSDFensemble.num_types)
        net_in = net_in[perm]
        out_sorted, color_communication = self.deepSDFensemble(net_in, anchor_idx_j_sorted)

        if self.color_branch and not skip_color:
            #net_in_color =  torch.cat([glob_feats_j, color_feats_j], dim=-1)[perm]
            net_in_color =  color_feats_j[perm]
            if glob_feats_color_j is not None:
                net_in_color = torch.cat([glob_feats_color_j[perm], net_in_color], dim=-1)
            #color_communication = self.layernorm(color_communication)
            _delta = delta[perm]
            if self.num_freq_bands is not None and self.num_freq_bands > 0:
                delta_embeds = [_delta]
                for freq in self.freq_bands:
                    delta_embeds.append(torch.sin(_delta * freq))
                    delta_embeds.append(torch.cos(_delta * freq))

                delta_embeds = torch.cat(delta_embeds, dim=-1)
            else:
                delta_embeds = _delta
            if self.color_communication:
                #net_in_color = torch.cat([net_in_color, color_communication, delta_embeds], dim=-1)
                color_communication = self.communcation_bottleneck1(color_communication)#, anchor_idx_j_sorted)
                color_communication = self.activation_bottleneck(color_communication)
                color_communication = self.communcation_bottleneck2(color_communication) #, anchor_idx_j_sorted)
                net_in_color = torch.cat([net_in_color, color_communication], dim=-1)
            else:
                net_in_color = torch.cat([net_in_color, delta_embeds], dim=-1)

            #out_color, _ = self.app_ensemble(net_in_color, anchor_idx_j_sorted)
            out_color, _ = self.aggr_app(net_in_color, anchor_idx_j_sorted)

        # undo sorting
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        out = out_sorted[inv, :]
        if self.color_branch and not skip_color:
            out_color = out_color[inv, :]

        _distances = torch.norm(delta[..., :3], dim=-1)
        distances = torch.where(anchor_idx_j  == (self.n_anchors-self.n_symm), 0.75, _distances)
        alpha = softmax_mod_varySize2(distances, index, ptr, num_neighbors=self.num_neighbors, blend_std=self.blend_std)
        alpha_color = alpha.unsqueeze(-1)

        if debug_region:
            self.latest_wights = alpha

        if self.color_branch and not skip_color:
            return torch.cat([out * alpha.unsqueeze(-1),  out_color*alpha_color], dim=-1)
        else:
            return alpha.unsqueeze(-1) * out

    def get_anchors(self, lat_rep_geo):

        glob_feats = lat_rep_geo[:, 0, :self.lat_dim_glob]
        offsets = self.mlp_pos(glob_feats).view(-1, self.n_anchors, 3)
        return self.anchors.to(offsets.device) + offsets


    def get_reg_loss(self, latent_code):
        loss_dict = {}

        loss_dict['reg_loc_geo'] = (torch.norm(latent_code['geo'][..., self.lat_dim_glob:], dim=-1) ** 2).mean()
        loss_dict['reg_global_geo'] = (torch.norm(latent_code['geo'][..., :self.lat_dim_glob], dim=-1) ** 2).mean()

        loss_dict['reg_loc_app'] = (torch.norm(latent_code['app'][..., self.lat_dim_glob:], dim=-1) ** 2).mean()
        loss_dict['reg_global_app'] = (torch.norm(latent_code['app'][..., :self.lat_dim_glob], dim=-1) ** 2).mean()

        symm_dist_geo, middle_dist_geo = self.get_symm_reg(latent_code['geo'].squeeze(1), 'geo')
        symm_dist_app, middle_dist_app = self.get_symm_reg(latent_code['app'].squeeze(1), 'app')

        loss_dict['symm_dist_geo'] = symm_dist_geo
        loss_dict['symm_dist_app'] = symm_dist_app

        return loss_dict


    def get_symm_reg(self, cond, cond_type : str):
        if cond_type == 'geo':
            shape_dim_glob = self.lat_dim_glob
            shape_dim_loc = self.lat_dim_loc_geo
        elif cond_type == 'app':
            shape_dim_glob = self.lat_dim_glob
            shape_dim_loc = self.lat_dim_loc_app
        n_symm = self.num_symm_pairs
        loc_lats_symm = cond[:, shape_dim_glob:shape_dim_glob + 2 * n_symm * shape_dim_loc].view(
            cond.shape[0], n_symm * 2, shape_dim_loc)
        loc_lats_middle = cond[:, shape_dim_glob + 2 * n_symm * shape_dim_loc:-shape_dim_loc].view(
            cond.shape[0], self.num_kps - n_symm * 2, shape_dim_loc)

        symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
        if loc_lats_middle.shape[1] % 2 == 0:
            middle_dist = torch.norm(loc_lats_middle[:, ::2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        else:
            middle_dist = torch.norm(loc_lats_middle[:, :-1:2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        return symm_dist, middle_dist


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')



def get_id_model(cfg,
                 spatial_input_dim,
                 include_color_branch=True,
                 rank=None,
                 ):
    '''
    Instantiate NPHM model from configs.
    '''

    cfg_id = cfg['id']
    device = 0
    anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH.format(cfg_id['nloc']))).float().unsqueeze(
        0).unsqueeze(0).to(device)


    print('ANCHORS HAVE SHAPE: ', anchors.shape)

    # TODO
    #assert cfg_id['lat_dim_loc_geo'] == cfg_id['lat_dim_loc_app']
    id_model = NPHM(
        lat_dim_glob=cfg_id['lat_dim_glob'],
        lat_dim_loc=cfg_id['lat_dim_loc_geo'] + cfg_id['lat_dim_loc_app'],
        hidden_dim_geo=cfg_id['gnn']['hidden_dim_geo'],
        hidden_dim_app=cfg_id['gnn']['hidden_dim_app'],
        n_anchors=cfg_id['nloc'],
        n_symm=cfg_id['nsymm_pairs'],
        anchors=anchors,
        n_layers_geo=cfg_id['gnn']['nlayers_geo'],
        n_layers_app=cfg_id['gnn']['nlayers_app'],
        pos_mlp_dim=64,
        num_neighbors=cfg_id['nneigh'],
        num_freq_bands=0,
        color_branch=include_color_branch,
        color_communication=True,
        d_pos=spatial_input_dim,
        rank=rank,
        is_monolith=False,
        blend_std=cfg_id['blend_std'],
        )

    return id_model#, anchors


