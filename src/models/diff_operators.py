import torch
from torch.autograd import grad


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def jac(decoder_expr, in_dict):

    in_dict['queries'].requires_grad_(True)

    out_dict = decoder_expr(in_dict)
    posed = in_dict['queries'] + out_dict['offsets']

    grads = []
    for i in range(posed.shape[-1]):
        d_out = torch.zeros_like(posed, requires_grad=False)
        d_out[:, :, i] = 1
        grad = torch.autograd.grad(
            outputs=posed,
            inputs=in_dict['queries'],
            grad_outputs=d_out,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads.append(grad)

    return torch.stack(grads, dim=-2)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradientSIREN(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def gradient(outputs, inputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True)[0][:, :, -3:]
    return points_grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status




