import sys
import zuko
import torch
import shutil
import logging
import sklearn
import sigkernel
import numpy as np
import pandas as pd
from time import time, sleep
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict
import matplotlib.pyplot as plt

from .plotting import plot_posterior
from .torch_jacfwd import jacfwd
from .sgld import SGLD
from .mpi_setup import mpi_comm, mpi_rank, mpi_size


def _setup_optimizer(model, flow, learning_rate, n_epochs):
    parameters_to_optimize = list(flow.parameters())
    #optimizer = SGLD(parameters_to_optimize, lr=learning_rate, noise_scaling=1.0)
    optimizer = torch.optim.AdamW(parameters_to_optimize, lr=learning_rate)
    #optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate, momentum=0.9)
    scheduler = None # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    n_parameters = sum([len(a) for a in parameters_to_optimize])
    print(f"Training flow with {n_parameters} parameters")
    return optimizer, scheduler


def _setup_loss(loss_name):
    loss_fn = torch.nn.MSELoss(reduction="mean")
    if loss_name == "LogMSELoss":
        def loss(x, y):
            mask = (x > 0) & (y > 0)  # remove points where log is not defined.
            return loss_fn(torch.log10(x[mask]), torch.log10(y[mask]))
    elif loss_name == "MSELoss":
        loss = lambda x, y: loss_fn(x, y)
    elif loss_name == "RelativeError":
        def loss(x, y):
            mask = y > 0
            return loss_fn(x[mask] / y[mask], y[mask] / y[mask])
    else:
        raise ValueError("Loss not supported")
    return loss


def _setup_paths(save_dir):
    save_dir = Path(save_dir)
    try:
        shutil.rmtree(save_dir)
    except:
        pass
    posteriors_dir = save_dir / "posteriors"
    models_dir = save_dir / "saved_models"
    posteriors_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    return save_dir, posteriors_dir, models_dir

def _compute_forecast_loss_reverse(
    model, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    loss = 0.0
    for i in range(n_samples):
        params = flow_cond.rsample()
        outputs_list = model(params)
        loss_i = 0.0
        try:
            assert len(outputs_list) == len(obs_data)
        except:
            raise ValueError(
                "Model results should be the same length as observed data."
            )
        for j in range(len(outputs_list)):
            observation = obs_data[j]
            model_output = outputs_list[j]
            loss_j = loss_fn(observation, model_output)
            if torch.isnan(loss_j):
                continue
            loss_i += loss_j
        loss += loss_i
    return loss / n_samples

def _compute_forecast_loss_reverse_score(
    model, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    loss = 0.0
    to_backprop = 0.0
    for i in range(n_samples):
        params = flow_cond.rsample()
        sample_log_prob = flow_cond.log_prob(params)
        outputs_list = model(params.detach())
        loss_i = 0.0
        try:
            assert len(outputs_list) == len(obs_data)
        except:
            raise ValueError(
                "Model results should be the same length as observed data."
            )
        for j in range(len(outputs_list)):
            observation = obs_data[j]
            model_output = outputs_list[j]
            loss_j = loss_fn(observation, model_output)
            if torch.isnan(loss_j):
                continue
            loss_i += loss_j
        loss += loss_i
        to_backprop += loss_i * sample_log_prob / n_samples
    to_backprop.backward()
    return loss / n_samples

def _compute_forecast_loss_forward(
    model, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    def aux_fun(params):
        loss = 0.0
        outputs_list = model(params)
        try:
            assert len(outputs_list) == len(obs_data)
        except:
            raise ValueError(
                "Model results should be the same length as observed data."
            )
        for i in range(len(outputs_list)):
            observation = obs_data[i]
            model_output = outputs_list[i]
            loss += loss_fn(observation, model_output)
        loss = loss / n_samples
        return loss, loss  # first is diffed, second is not.

    # Sample from flow
    if mpi_rank == 0:
        params_list = flow_cond.rsample((n_samples,))
        params_list_comm = params_list.detach().cpu().numpy()
    else:
        params_list_comm = None
    # scatter params to ranks
    if mpi_comm is not None:
        params_list_comm = mpi_comm.bcast(params_list_comm, root=0)
    # Compute ABM jacobian with Forward-diff
    jac_f = jacfwd(
        aux_fun, 0, randomness="same", has_aux=True, chunk_size=jacobian_chunk_size
    )
    total_loss = 0
    total_params_diff = 0
    jacobians_per_rank = []
    parameters_indices_per_rank = []
    for i, params_comm in enumerate(params_list_comm):
        if i % mpi_size == mpi_rank:
            parameters_indices_per_rank.append(i)
            jacobian, loss = jac_f(
                torch.tensor(params_comm, device=model.device)
            )  # Important to detach here since we don't reverse diff this.
            total_loss += loss
            jacobians_per_rank.append(jacobian.cpu().numpy())
    jacobians_per_rank = np.array(jacobians_per_rank)
    if mpi_comm is not None:
        jacobians_comm = mpi_comm.gather(jacobians_per_rank, root=0)
    else:
        jacobians_comm = jacobians_per_rank
    if mpi_comm is not None:
        parameters_indices_per_rank = mpi_comm.gather(parameters_indices_per_rank, root=0)
        total_loss = np.array(mpi_comm.gather(total_loss.item(), root = 0))
    else:
        total_loss = total_loss.cpu().numpy()
    if mpi_rank == 0:
        parameters_indices = torch.tensor([i for i_rank in parameters_indices_per_rank for i in i_rank])
        parameters_ordered = params_list[parameters_indices]
        jacobians_unrolled = [jacobian for jacobian_comm in jacobians_comm for jacobian in jacobian_comm]
        jacobians_unrolled = torch.tensor(np.stack(jacobians_unrolled), device=model.device, dtype=torch.float)
        total_params_diff = 0.0
        n_samples_non_nan = 0
        for i in range(n_samples):
            jacobian = jacobians_unrolled[i,:]
            parameters = parameters_ordered[i,:]
            if torch.isnan(jacobian).any():
                continue
            # Use reverse diff for flow
            total_params_diff += torch.dot(jacobian, parameters)
            n_samples_non_nan += 1
        # Back-propagate to flow parameters
        total_params_diff = total_params_diff / n_samples_non_nan
        total_params_diff.backward()
        return torch.tensor(sum(total_loss[~np.isnan(total_loss)]) / n_samples_non_nan)

def _compute_signature_kernel_sigma(time_series):
    pairwise_distances = sklearn.metrics.pairwise_distances(time_series.reshape(-1,1).cpu().numpy())
    sigma = np.median(pairwise_distances)
    return sigma

def _compute_forecast_loss_signature_kernel_reverse(
    model, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    time_index = torch.linspace(0, 1, obs_data[0].shape[0], device = obs_data[0].device)
    y = torch.vstack((time_index, torch.log10(obs_data[0]))).transpose(0,1)[None,:].to(torch.double)
    Xs = []
    time_index_batched = time_index.repeat(n_samples).reshape(n_samples, -1)
    for i in range(n_samples):
        params = flow_cond.rsample()
        outputs = torch.hstack(model(params))
        Xs.append(outputs)
    X = torch.log10(torch.vstack(Xs))
    X = torch.cat((time_index_batched[:,None], X[:,None]), 1).transpose(1,2).to(torch.double)
    #Y = y.repeat((X.shape[0],1,1))
    #assert X.shape == (n_samples, obs_data[0].shape[0], 2)
    #assert Y.shape == X.shape
    #score_xy = loss_fn.compute_kernel(X, Y)
    #score_yy = loss_fn.compute_kernel(Y, Y)
    #loss = torch.nn.MSELoss(reduction="mean")(score_xy, score_yy)
    loss = loss_fn.compute_scoring_rule(X, y) + loss_fn.compute_kernel(y,y,1)
    print(loss)
    return loss

def _get_regularisation(flow_cond, prior, n_samples=5):
    """
    Computes the KL divergence between the flow and the prior
    """
    samples = flow_cond.rsample((n_samples,))
    flow_lps = flow_cond.log_prob(samples)
    prior_lps = prior.log_prob(samples)
    kl = torch.mean(flow_lps - prior_lps)
    return kl


def _plot_posterior(flow_cond, true_values, posteriors_dir, it, **kwargs):
    f = plot_posterior(flow_cond=flow_cond, true_values=true_values, **kwargs)
    f.savefig(posteriors_dir / f"posterior_{it:03d}.png", dpi=50, bbox_inches="tight")
    plt.close(f)


def infer(
    model: torch.nn.Module,
    flow: zuko.flows.FlowModule,
    prior: torch.distributions.Distribution,
    obs_data: list[torch.Tensor],
    diff_mode="reverse",
    gradient_estimation_mode="pathwise",
    jacobian_chunk_size: int = None,
    n_epochs: int = 100,
    n_samples_per_epoch: int = 10,
    n_samples_regularization: int = 1000,
    w: float = 0.5,
    save_dir: str = "./results",
    learning_rate: float = 1e-3,
    loss: str = "LogMSELoss",
    true_values: Optional[list] = None,
    plot_posteriors: str = "every",
    device="cpu",
    preload_model_path: str = None,
    progress_bar: bool = True,
    **kwargs,
):
    r"""
    Runs simulation-based inference on the given model and data using the specified flow and prior.

    Parameters
    ----------
    model:
        The model to do inference on. Needs to be specified as a `torch.nn.Module`. The forward first argument should be the array of predicted parameters. Any additional kwargs passed to this function will be passed to the `forward` pass of the model.
        The parameters to calibrate are assumed to be the variables in the model specified as `torch.nn.Parameter`, ie, the ones showing in `list(model.parameters())`.
    flow:
        A zuko NF. Example:
            ```python
            flow = zuko.flows.NSF(4, 1, transforms=3, hidden_features=[128] * 3)
            ```
    prior:
        A distribution object in PyTorch. Example:
            ```python
            prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(4, device=device),
                covariance_matrix=torch.eye(len(params_to_calibrate), device=device),
            )
            ```
    obs_data: A list of `torch.Tensor` with the observed data series. The shapes should correspond exactly to the output of the `forward()` method of the simulator.
    diff_mode: which diff mode to use. Options are "reverse" or "forward". For high memory tasks use "forward", "reverse" is faster.
    gradient_estimation_mode: Gradient estimation mode. Options are "pathwise" and "score".
    jacobian_chunk_size: chunk size for torch vmap to compute jacobian. Set to None for max perforamnce, or low number when out of memory.
    n_epochs: Number of epochs
    n_samples_per_epoch: Number of sets of parameters sampled from the flow for each epoch.
    n_samples_regularization: Number of samples used to calculate the regularization term between the flow and the prior.
    w: Regularization weight. Higher values give more importance to the prior.
    save_dir: Path to save results.
    true_values: (Optional) If known, true parameter values of the generating model. Will be shown in temporary plots.
    device: what device to use (cpu or cuda:0 etc.)
    plot_posteriors: When to save the posteriors. Options: "best" : only save when loss is improved, "every" save all, "never" : never save.
    **kwargs: Keyword arguments to be passed to model.
    """
    if preload_model_path:
        flow.load_state_dict(torch.load(preload_model_path, map_location=device))
    optimizer, scheduler = _setup_optimizer(model, flow, learning_rate, n_epochs)
    save_dir, posteriors_dir, models_dir = _setup_paths(save_dir)
    w = torch.tensor(w, requires_grad=True)
    losses = defaultdict(list)
    best_loss = np.inf
    best_forecast_loss = np.inf
    if loss == "signature_kernel":
        sigma = _compute_signature_kernel_sigma(obs_data[0]) #TODO: expand to any dim.
        static_kernel = sigkernel.RBFKernel(sigma=sigma)
        loss_fn = sigkernel.SigKernel(static_kernel, dyadic_order=1)
        forecast_loss_fn = _compute_forecast_loss_signature_kernel_reverse
    else:
        loss_fn = _setup_loss(loss)
        if gradient_estimation_mode == "pathwise":
            if diff_mode == "reverse":
                forecast_loss_fn = _compute_forecast_loss_reverse
            elif diff_mode == "forward":
                forecast_loss_fn = _compute_forecast_loss_forward
            else:
                raise ValuError(f"Diff mode {diff_mode} not supported.")
        elif gradient_estimation_mode == "score":
            if diff_mode == "reverse":
                forecast_loss_fn = _compute_forecast_loss_reverse_score
            else:
                raise ValuError(f"Diff mode {diff_mode} not supported with score gradient estimation.")
        else:
            raise ValueError(f"Gradient estimation mode {gradient_estimation_mode} not supported.")
    if mpi_rank == 0 and progress_bar:
        iterator = tqdm(range(n_epochs))
    else:
        iterator = range(n_epochs)
    for it in iterator:
        if mpi_rank == 0:
            need_to_plot_posterior = plot_posteriors == "every"
            flow_cond = flow(torch.zeros(1, device=device))
            optimizer.zero_grad()
        else:
            flow_cond = None
        forecast_loss = forecast_loss_fn(
            model=model,
            flow_cond=flow_cond,
            obs_data=obs_data,
            loss_fn=loss_fn,
            n_samples=n_samples_per_epoch,
            jacobian_chunk_size=jacobian_chunk_size,
        )
        if mpi_rank == 0:
            reglrise_loss = w * _get_regularisation(
                flow_cond=flow_cond, prior=prior, n_samples=n_samples_regularization
            )
            loss = forecast_loss + reglrise_loss
            losses["forecast"].append(forecast_loss.item())
            losses["reglrise"].append(reglrise_loss.item())
            losses["total"].append(loss.item())
            if torch.isnan(loss):
                torch.save(flow.state_dict(), models_dir / f"to_debug.pth")
                raise ValueError("Loss is nan!")
            if diff_mode == "reverse" and gradient_estimation_mode == "pathwise":
                loss.backward()
            else:
                reglrise_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            torch.save(flow.state_dict(), models_dir / f"model_{it:04d}.pth")
            if loss.item() < best_loss:
                torch.save(flow.state_dict(), models_dir / f"best_model_{it:04d}.pth")
                best_loss = loss.item()
                need_to_plot_posterior = need_to_plot_posterior or (
                    plot_posteriors == "best"
                )
            if forecast_loss.item() < best_forecast_loss:
                torch.save(
                    flow.state_dict(), models_dir / f"best_model_forecast_{it:04d}.pth"
                )
                best_forecast_loss = forecast_loss.item()
                need_to_plot_posterior = need_to_plot_posterior or (
                    plot_posteriors == "best"
                )
            df = pd.DataFrame(losses)
            df.to_csv(save_dir / "losses_data.csv", index=False)
            if need_to_plot_posterior:
                _plot_posterior(
                    flow_cond=flow_cond,
                    true_values=true_values,
                    posteriors_dir=posteriors_dir,
                    it=it,
                    **kwargs,
                )
            optimizer.step()
            #scheduler.step()
