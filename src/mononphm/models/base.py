import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Optional, Literal, List


class IdModel(nn.Module, ABC):
    def __init__(self):
        super(IdModel, self).__init__()

    @abstractmethod
    def forward(self,
                queries: torch.Tensor,
                lat_id : torch.Tensor):
        pass


class ExModel(nn.Module, ABC):
    def __init__(self):
        super(ExModel, self).__init__()

    @abstractmethod
    def forward(self,
                queries: torch.Tensor,
                lat_ex : torch.Tensor,
                lat_id : Optional[torch.Tensor]):
        pass


class LatentCodes(nn.Module):

    def __init__(self,
                 n_latents : List[int],
                 n_channels : List[int],
                 modalities : List[Literal['geo', 'app', 'exp']],
                 types : List[Literal['vector', 'grid', 'triplane']],
                 resolutions : Optional[List[Optional[int]]] = None,
                 init_zeros : bool = False,
                 variational : bool = False,
                 ):
        super(LatentCodes, self).__init__()

        self.variational = variational
        self.codebook = torch.nn.ModuleDict()
        for i, mod in enumerate(modalities):
            self.codebook[mod] = SingleModalityLatentCodes(n_latents[i],
                                                           n_channels[i],
                                                           resolutions[i] if resolutions is not None else None,
                                                           types[i],
                                                           init_zeros=init_zeros,
                                                           )
        if self.variational:
            self.codebook_logvar = torch.nn.ModuleDict()

            for i, mod in enumerate(modalities):
                self.codebook_logvar[mod] = SingleModalityLatentCodes(n_latents[i],
                                                                       n_channels[i],
                                                                       resolutions[i] if resolutions is not None else None,
                                                                       types[i],
                                                                       init_zeros=init_zeros,
                                                                       )

    def forward(self, latent_idx, return_mu_sig=False):
        if self.variational:
            code_dict  = {}
            mu_dict = {}
            log_var_dict = {}
            for mod in self.codebook.keys():
                log_var = self.codebook_logvar[mod](latent_idx[mod])
                std = torch.exp(0.5 * log_var) #+ 1e-8
                eps = torch.randn_like(std)
                mu = self.codebook[mod](latent_idx[mod])
                code_dict[mod] = eps * std + mu
                mu_dict[mod] = mu
                log_var_dict[mod] = log_var
            if return_mu_sig:
                return code_dict, mu_dict, log_var_dict
            else:
                return code_dict
        else:
            code_dict =  {mod: codes(latent_idx[mod]) for (mod, codes) in self.codebook.items()}
        return code_dict

    #def get_all_params(self):
    #    all_params = []
    #    for mod in self.codebook.keys():
    #        all_params += self.codebook[mod].parameters()
    #    return all_params


class SingleModalityLatentCodes(nn.Module):
    def __init__(self,
                 n_latents : int,
                 n_channels : int,
                 resolution : Optional[int] = None,
                 type : Literal['vector', 'grid', 'triplane'] = 'vector',
                 init_zeros : bool = False):
        super(SingleModalityLatentCodes, self).__init__()

        if type in ['grid', 'triplane']:
            assert resolution is not None and resolution > 0
        else:
            assert resolution is None

        if type == 'vector':
            dim = n_channels
        elif type == 'grid':
            dim = resolution ** 3
        elif type == 'triplane':
            dim = 3*resolution**2
        else:
            raise ValueError(f'Unexpected value for latent type encountered: {type}!')

        self.embedding = torch.nn.Embedding(n_latents, dim,
                           max_norm=1.0, sparse=True, device='cuda').float()

        if init_zeros:
            torch.nn.init.zeros_(
                self.embedding.weight.data,
            )
        else:
            torch.nn.init.normal_(
                self.embedding.weight.data,
                0.0,
                0.001,
            )

    def forward(self, input_idx):
        return self.embedding(input_idx)
