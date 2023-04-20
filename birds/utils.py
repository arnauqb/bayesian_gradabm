import numpy as np
import torch
import torch.autograd as autograd
import random

def fix_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Fixing seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _estimate_w(
    model, flow_cond, prior, obs_data, loss_fn, N, M
):

    numerator = 0.
    denominator = 0.
    for n in range(N):
        params = prior.rsample(1)
        params.requires_grad_(True)
        out = model(params)
        loss = loss_fn(obs_data[0], out[0])
        gradl = autograd.grad(loss, params)[0]
        print("")
        print("den", loss.item(), gradl)
        print("den gradl sq", torch.pow(gradl, 2).sum())
        print("")
        denominator += torch.pow(gradl, 2).sum()
        for m in range(M):
            params = prior.rsample(1)
            params.requires_grad_(True)
            out = model(params)
            outputs_list = model(flow_cond.rsample())
            loss = loss_fn(outputs_list[0], out[0])
            gradl = autograd.grad(loss, params)[0]
            print("num", loss.item(), gradl)
            print("num gradl sq", torch.pow(gradl, 2).sum())
            numerator += torch.pow(gradl, 2).sum()
    numerator, denominator = numerator / (N*M), denominator / N
    return torch.sqrt(numerator / denominator)
            

def __estimate_w(
    model, flow_cond, obs_data, loss_fn, N
):

    J = 0.
    I = 0.
    fun = lambda params: loss_fn(obs_data[0], model(params)[0])
    for n in range(N):
        params = flow_cond.rsample()
        outputs_list = model(params)
        loss = loss_fn(obs_data[0], outputs_list[0])
        first_derivative = autograd.grad(loss, params, create_graph=True)[0]
        I += torch.outer(first_derivative, first_derivative)
        second_derivative = autograd.functional.hessian(fun, params)
        J += second_derivative
    J, I = J / N, I / N
    trJ = torch.trace(J)
    trJIJ = torch.trace( J.dot(torch.inverse(I).dot(torch.transpose(J, -2, -1))) )
    return trJIJ / trJ
