import torch
import torch.nn as nn

class WrapMonoNPHM(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.monoNPHM = model


    def forward(self, positions, condition, include_color : bool = False, return_grad : bool = False):

        if len(positions.shape) == 2:
            positions = positions.unsqueeze(0)
        result = self.monoNPHM({'queries': positions}, cond=condition, skip_color=not include_color, return_grad=return_grad)
        sdf = result['sdf'].squeeze(0)

        if include_color:
            color = result['color'].squeeze(0)
            #sdf, _, color = self.GTA(positions, self.latent_code, None, squeeze=True)
            if return_grad:
                return torch.cat([sdf, color, result['gradient'].squeeze(0)], dim=-1)
            else:
                return torch.cat([sdf, color], dim=-1)
        else:
            return sdf

    def warp(self, positions, expression):
        condition = self.latent_code_id
        condition.update({'exp': self.latent_codes_expr(torch.tensor([expression], device=positions.device)).unsqueeze(0)})

        in_dict = {'queries': positions, 'cond': condition}

        if hasattr(self.monoNPHM.id_model, 'mlp_pos') and self.monoNPHM.id_model.mlp_pos is not None and 'anchors' not in in_dict:
            in_dict.update({'anchors': self.monoNPHM.id_model.get_anchors(condition['geo'])})
        out_ex = self.monoNPHM.ex_model(in_dict)
        queries_canonical = in_dict['queries'] + out_ex['offsets']


        return queries_canonical

    def gradient(self, x, expression):
        x.requires_grad_(True)
        condition = self.latent_code_id
        condition.update({'exp': self.latent_codes_expr(torch.tensor([expression], device=x.device)).unsqueeze(0)})
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        result = self.monoNPHM.forward({'queries': x}, condition, None)
        y = result['sdf']
        y = y.squeeze(0)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0].squeeze(0)
        return gradients.unsqueeze(1)
