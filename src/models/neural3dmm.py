import torch
from torch import nn
from typing import Optional, Literal
from inspect import getfullargspec

from mononphm.models.base import IdModel, ExModel
from mononphm.models.diff_operators import gradient


class n3dmm():
    def __init__(self,
                 id_model : IdModel,
                 ex_model : ExModel,
                 expr_direction : Optional[Literal['forward', 'backward', 'monolith']],
                 neutral_only : bool = False):
        super().__init__()

        self.id_model = id_model
        self.ex_model = ex_model
        self.expr_direction = expr_direction
        self.neutral_only =neutral_only


    def __call__(self,
                in_dict,
                cond,
                return_grad : bool = False,
                 skip_color : bool = False,
                 no_grad_mask = None,
                 ignore_deformations = False # unused
                 ):
        if return_grad:
            in_dict['queries'].requires_grad_()
        in_dict.update({'cond': cond})
        #TODO
        if self.expr_direction == 'forward':
            pass
        elif self.expr_direction == 'backward':
            # provide expressions with predicted facial anchors
            if hasattr(self.id_model, 'mlp_pos') and self.id_model.mlp_pos is not None and 'anchors' not in in_dict:
                in_dict.update({'anchors': self.id_model.get_anchors(cond['geo'])})
            if not self.neutral_only:
                #with torch.no_grad():
                if no_grad_mask is not None:
                    out_ex_grad = self.ex_model({'annchors': in_dict['anchors'], 'cond': in_dict['cond'], 'queries': in_dict['queries'][~no_grad_mask, ...]})
                    with torch.no_grad():
                        out_ex_no_grad = self.ex_model({'annchors': in_dict['anchors'], 'cond': in_dict['cond'], 'queries': in_dict['queries'][no_grad_mask, ...]})
                    out_ex = {k: torch.where(no_grad_mask, out_ex_no_grad[k], out_ex_grad[k]) for k in out_ex_grad.keys()}
                else:
                    out_ex = self.ex_model(in_dict)
                queries_canonical = in_dict['queries'] + out_ex['offsets']
                if self.ex_model.n_hyper > 0:
                    queries_canonical = torch.cat([queries_canonical, out_ex['hyper_coords']], dim=-1)
            else:
                out_ex = {'offsets': torch.zeros_like(in_dict['queries'])}
                queries_canonical = in_dict['queries']
            in_dict.update({'queries_can': queries_canonical, 'offsets': out_ex['offsets']})
            pred = self.id_model(in_dict, skip_color=skip_color)
            if self.ex_model.sdf_corrective:
                pred['sdf'] += out_ex['sdf_corrective'] # TODO how to enforce eikonal constraint exactly??
                pred['sdf_corrective'] = out_ex['sdf_corrective']
            if self.ex_model.n_hyper > 0:
                pred['hyper_coords'] = out_ex['hyper_coords']
            if return_grad:
                grad = gradient(pred['sdf'], in_dict['queries'])
                pred.update({'gradient': grad})
            pred.update({'offsets': out_ex['offsets']})
            return pred
        #TODO
        elif self.expr_direction == 'monolith':
            return self.ex_model(queries, lat_ex, lat_id)
        #TODO
        elif self.expr_direction is None:
            assert lat_ex is None
        else:
            raise ValueError(f'unexpected value for {self.expr_direction}')
        assert 1 == 2


class nn3dmm(nn.Module):
    def __init__(self,
                 id_model : IdModel,
                 ex_model : ExModel,
                 expr_direction : Optional[Literal['forward', 'backward', 'monolith']],
                 neutral_only : bool = False):
        super().__init__()

        self.id_model = id_model
        self.ex_model = ex_model
        self.expr_direction = expr_direction
        self.neutral_only =neutral_only


    def forward(self,
                in_dict,
                cond,
                return_grad : bool = False,
                return_can : bool = False,
                skip_color : bool = False,
                ignore_deformations : bool = False,
                ):
        if return_grad:
            in_dict['queries'].requires_grad_()
        in_dict.update({'cond': cond})
        #TODO
        if self.expr_direction == 'forward':
            pass
        elif self.expr_direction == 'backward':
            # provide expressions with predicted facial anchors
            if hasattr(self.id_model, 'mlp_pos') and self.id_model.mlp_pos is not None and 'anchors' not in in_dict:
                in_dict.update({'anchors': self.id_model.get_anchors(cond['geo'])})
            if not self.neutral_only:
                if ignore_deformations:
                    out_ex = self.ex_model(in_dict)
                    queries_canonical = in_dict['queries']
                    if self.ex_model.n_hyper > 0:
                        queries_canonical = torch.cat([queries_canonical, torch.zeros_like(out_ex['hyper_coords'])], dim=-1)
                else:
                    out_ex = self.ex_model(in_dict)
                    queries_canonical = in_dict['queries'] + out_ex['offsets']
                    if self.ex_model.n_hyper > 0:
                        queries_canonical = torch.cat([queries_canonical, out_ex['hyper_coords']], dim=-1)
            else:
                out_ex = {'offsets': torch.zeros_like(in_dict['queries'])}
                queries_canonical = in_dict['queries']
            in_dict.update({'queries_can': queries_canonical, 'offsets': out_ex['offsets']})
            if skip_color in getfullargspec(self.id_model.forward)[0]:
                pred = self.id_model(in_dict, skip_color=skip_color)
            else:
                pred = self.id_model(in_dict)
            if self.ex_model.sdf_corrective:
                pred['sdf'] += out_ex['sdf_corrective'] # TODO how to enforce eikonal constraint exactly??
                pred['sdf_corrective'] = out_ex['sdf_corrective']
            if self.ex_model.n_hyper > 0:
                pred['hyper_coords'] = out_ex['hyper_coords']
            if return_grad:
                grad = gradient(pred['sdf'], in_dict['queries'])
                pred.update({'gradient': grad})
            pred.update({'offsets': out_ex['offsets']})
            if return_can:
                pred.update({'queries_can': queries_canonical})
            return pred
        #TODO
        elif self.expr_direction == 'monolith':
            return self.ex_model(queries, lat_ex, lat_id)
        #TODO
        elif self.expr_direction is None:
            assert lat_ex is None
        else:
            raise ValueError(f'unexpected value for {self.expr_direction}')
        assert 1 == 2
