import sys
import zuko
import torch
import shutil
import logging
import sklearn
import sklearn.metrics
import sigkernel
import numpy as np
import pandas as pd
from time import time, sleep
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Optional, List, Callable

from .plotting import plot_posterior
from .torch_jacfwd import jacfwd
from .sgld import SGLD
from .mpi_setup import mpi_comm, mpi_rank, mpi_size


def _setup_optimizer(model: torch.nn.Module, flow: torch.nn.Module, learning_rate: float, n_epochs: int) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Sets up the optimizer and learning rate scheduler for training a given flow model.

    Args:
        model (torch.nn.Module): The model to be trained.
        flow (torch.nn.Module): The flow model to be optimized.
        learning_rate (float): The learning rate for the optimizer.
        n_epochs (int): The number of epochs to train the model for.

    Returns:
        Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]: A tuple containing the optimizer and learning rate scheduler.

    Raises:
        TypeError: If the model or flow arguments are not of type torch.nn.Module.
        ValueError: If the learning_rate argument is not a positive float or the n_epochs argument is not a positive integer.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model argument must be of type torch.nn.Module.")
    if not isinstance(flow, torch.nn.Module):
        raise TypeError("The flow argument must be of type torch.nn.Module.")
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        raise ValueError("The learning_rate argument must be a positive float.")
    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError("The n_epochs argument must be a positive integer.")

    parameters_to_optimize = list(flow.parameters())
    optimizer = torch.optim.AdamW(parameters_to_optimize, lr=learning_rate)
    scheduler = None
    n_parameters = sum([len(a) for a in parameters_to_optimize])
    print(f"Training flow with {n_parameters} parameters")
    return optimizer, scheduler


def _setup_loss(loss_name):
    if callable(loss_name):
        return loss_name
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


def _setup_paths(save_dir: str) -> Tuple[Path, Path, Path]:
    """
    Sets up the necessary directories for saving posteriors and models.

    Args:
        save_dir (str): The directory where the posteriors and models will be saved.

    Returns:
        Tuple[Path, Path, Path]: A tuple containing the save directory, the directory for posteriors, and the directory for saved models.

    Raises:
        OSError: If there is an error removing the save directory.
    """
    save_dir = Path(save_dir)
    try:
        shutil.rmtree(save_dir)
    except OSError as e:
        raise OSError(f"Error removing directory {save_dir}: {e}") from None
    posteriors_dir = save_dir / "posteriors"
    models_dir = save_dir / "saved_models"
    posteriors_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    return save_dir, posteriors_dir, models_dir


def _compute_forecast_loss_reverse(
    model, flow, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    loss = 0.0
    n_samples_not_nan = 0
    for _ in range(n_samples):
        params = flow_cond.rsample()
        if torch.isnan(params).any():
            warnings.warn("Flow generated nans")
        outputs_list = model(params)
        loss_i = 0.0
        try:
            assert len(outputs_list) == len(obs_data)
        except:
            raise ValueError(
                "Model results should be the same length as observed data."
            )
        for i in range(len(outputs_list)):
            observation = obs_data[i]
            model_output = outputs_list[i]
            if torch.isnan(model_output).any():
                warnings.warn("Simulation produced nan -- ignoring")
            loss_i_ = loss_fn(observation, model_output)
            if torch.isnan(loss_i_):
                continue
            n_samples_not_nan += 1
            loss_i += loss_i_
        loss += loss_i
    if n_samples_not_nan == 0:
        raise ValueError("All simulations had nans")
    return loss / n_samples_not_nan

def _compute_forecast_loss_reverse_two_step(
    model, flow, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
):
    loss = 0.0
    n_samples_not_nan = 0
    n_flow_parameters = len(list(flow.parameters()))
    gradients = torch.zeros(n_flow_parameters)
    for _ in range(n_samples):
        params = flow_cond.rsample()
        if torch.isnan(params).any():
            warnings.warn("Flow generated nans")
        params_for_abm = params.detach().clone()
        params_for_abm.requires_grad = True
        outputs_list = model(params_for_abm)
        loss_i = 0.0
        try:
            assert len(outputs_list) == len(obs_data)
        except:
            raise ValueError(
                "Model results should be the same length as observed data."
            )
        for i in range(len(outputs_list)):
            observation = obs_data[i]
            model_output = outputs_list[i]
            if torch.isnan(model_output).any():
                warnings.warn("Simulation produced nan -- ignoring")
            loss_i_ = loss_fn(observation, model_output)
            if torch.isnan(loss_i_):
                continue
            loss_i += loss_i_
        if torch.isnan(loss_i) or loss_i == 0.0:
            continue
        n_samples_not_nan += 1
        gradient_theta = torch.autograd.grad(loss_i, params_for_abm)
        gradients += torch.autograd.grad(params, [p for p in flow.parameters() if p.requires_grad], gradient_theta[0])
        loss += loss_i.item()
    # set gradients
    for k, param in enumerate(flow.parameters()):
        param.grad = gradients[k] / n_samples_not_nan
    if n_samples_not_nan == 0:
        raise ValueError("All simulations had nans")
    return loss / n_samples_not_nan

def _compute_forecast_loss_reverse_score(
    model: torch.nn.Module, 
    flow, 
    flow_cond: torch.distributions.Distribution, 
    obs_data: List[torch.Tensor], 
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    n_samples: int, 
    jacobian_chunk_size: int
) -> torch.Tensor:
    """
    Computes the reverse score of the forecast loss for a given model and observed data.

    Args:
        model (torch.nn.Module): The PyTorch model to use for computing the forecast loss.
        flow_cond (torch.distributions.Distribution): The distribution to use for sampling parameters for the model.
        obs_data (List[torch.Tensor]): A list of observed data to compare against the model's output.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use for comparing the model's output to the observed data.
        n_samples (int): The number of samples to use for computing the reverse score.
        jacobian_chunk_size (int): The chunk size to use for computing the Jacobian matrix.

    Returns:
        torch.Tensor: The computed reverse score of the forecast loss.

    Raises:
        ValueError: If the length of the model's output does not match the length of the observed data.

    """
    loss = torch.zeros(1, device=model.device, requires_grad=True)
    to_backprop = torch.zeros(1, device=model.device, requires_grad=True)
    n_samples_not_nan = 0
    for i in range(n_samples):
        params = flow_cond.sample()
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
            loss_i = loss_i + loss_j
            n_samples_not_nan += 1
        loss = loss + loss_i
        to_backprop = to_backprop + loss_i * sample_log_prob 
    to_backprop = to_backprop / n_samples_not_nan
    to_backprop.backward()
    return loss / n_samples


def _compute_forecast_loss_forward(
    model, flow, flow_cond, obs_data, loss_fn, n_samples, jacobian_chunk_size
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
        total_params_diff = torch.zeros(1, device=model.device, requires_grad=True)
        n_samples_non_nan = 0
        for i in range(n_samples):
            jacobian = jacobians_unrolled[i,:]
            parameters = parameters_ordered[i,:]
            if torch.isnan(jacobian).any():
                print("warning nan jacobian")
                continue
            # Use reverse diff for flow
            total_params_diff = total_params_diff + torch.dot(jacobian, parameters)
            n_samples_non_nan += 1
        # Back-propagate to flow parameters
        total_params_diff = total_params_diff / n_samples_non_nan
        total_params_diff.backward()
        return torch.tensor(sum(total_loss[~np.isnan(total_loss)]) / n_samples_non_nan)

def _compute_signature_kernel_sigma(time_series: np.ndarray) -> float:
    """
    Computes the Gaussian kernel scale parameter for the signature kernel using the median pairwise distance
    between time series.

    Parameters:
    -----------
    time_series : np.ndarray
        A 1D numpy array representing a time series.

    Returns:
    --------
    sigma : float
        The Gaussian kernel scale parameter for the signature kernel.

    Raises:
    -------
    None

    Notes:
    ------
    This function uses the scikit-learn library to compute the pairwise distances between time series. The
    resulting distances are used to compute the median, which is then returned as the Gaussian kernel scale
    parameter for the signature kernel. The function also prints a message indicating the value of sigma that
    is being used.
    """
    pairwise_distances = sklearn.metrics.pairwise_distances(time_series.reshape(-1,1).cpu().numpy())
    sigma = np.median(pairwise_distances)
    print("Using signature kernel with Gaussian kernel scale parameter {0}".format(sigma))
    return sigma

def _get_regularisation(flow_cond: torch.distributions.Distribution, prior: torch.distributions.Distribution, n_samples: int = 5) -> torch.Tensor:
    """
    Computes the KL divergence between the flow and the prior.

    Parameters:
    -----------
    flow_cond : torch.distributions.Distribution
        The distribution of the flow conditioned on the input.
    prior : torch.distributions.Distribution
        The prior distribution.
    n_samples : int, optional
        The number of samples to use for the computation. Default is 5.

    Returns:
    --------
    kl : torch.Tensor
        The KL divergence between the flow and the prior.

    Raises:
    -------
    TypeError:
        If `flow_cond` or `prior` are not instances of `torch.distributions.Distribution`.
    ValueError:
        If `n_samples` is not a positive integer.
    """

    if not isinstance(flow_cond, torch.distributions.Distribution) or not isinstance(prior, torch.distributions.Distribution):
        raise TypeError("Both `flow_cond` and `prior` must be instances of `torch.distributions.Distribution`.")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("`n_samples` must be a positive integer.")

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
    obs_data: List[torch.Tensor],
    diff_mode="reverse",
    gradient_estimation_mode="pathwise",
    jacobian_chunk_size: int = None,
    n_epochs: int = 100,
    n_samples_per_epoch: int = 10,
    n_samples_regularization: int = 1000,
    w: float = 0.5,
    clip_val: float = np.inf,
    save_dir: str = "./results",
    learning_rate: float = 1e-3,
    loss: str = "LogMSELoss",
    reg_loss: str = "KL",
    true_values: Optional[list] = None,
    plot_posteriors: str = "every",
    device="cpu",
    preload_model_path: str = None,
    progress_bar: bool = True,
    max_num_epochs_without_improvement: int = 20,
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
    loss_fn = _setup_loss(loss)
    if gradient_estimation_mode == "pathwise":
        if diff_mode == "reverse":
            forecast_loss_fn = _compute_forecast_loss_reverse
        elif diff_mode == "forward":
            forecast_loss_fn = _compute_forecast_loss_forward
        else:
            raise ValuError(f"Diff mode {diff_mode} not supported.")
    elif gradient_estimation_mode == "pathwise-parts":
        if diff_mode == "reverse":
            forecast_loss_fn = _compute_forecast_loss_reverse_two_step
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
    num_epochs_without_improvement = 0
    for it in iterator:
        if mpi_rank == 0:
            need_to_plot_posterior = plot_posteriors == "every"
            flow_cond = flow(torch.zeros(1, device=device))
            optimizer.zero_grad()
        else:
            flow_cond = None
        forecast_loss = forecast_loss_fn(
            model=model,
            flow=flow,
            flow_cond=flow_cond,
            obs_data=obs_data,
            loss_fn=loss_fn,
            n_samples=n_samples_per_epoch,
            jacobian_chunk_size=jacobian_chunk_size,
        )
        if mpi_rank == 0:
            if reg_loss == "KL":
                reglrise_loss = w * _get_regularisation(
                    flow_cond=flow_cond, prior=prior, n_samples=n_samples_regularization
                )
            else:
                reglrise_loss = w * reg_loss(flow_cond, prior, n_samples_regularization)
            loss = forecast_loss + reglrise_loss
            losses["forecast"].append(forecast_loss.item())
            losses["reglrise"].append(reglrise_loss.item())
            losses["total"].append(loss.item())
            if torch.isnan(loss):
                torch.save(flow.state_dict(), models_dir / f"to_debug.pth")
                raise ValueError("Loss is nan! Forecast {0} and reg {1}".format(forecast_loss.item(), reglrise_loss.item()))
            if diff_mode == "reverse" and gradient_estimation_mode == "pathwise":
                loss.backward()
            else:
                reglrise_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), clip_val)
            torch.save(flow.state_dict(), models_dir / f"model_{it:04d}.pth")
            if loss.item() < best_loss:
                torch.save(flow.state_dict(), models_dir / f"best_model_{it:04d}.pth")
                best_loss = loss.item()
                need_to_plot_posterior = need_to_plot_posterior or (
                    plot_posteriors == "best"
                )
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
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
        if num_epochs_without_improvement >= max_num_epochs_without_improvement:
            print("Max number of epochs without improvement reached")
            break
        tot_loss = forecast_loss.item() + reglrise_loss.item()
        iterator.set_postfix({"Forecast":forecast_loss.item(), "Reg.":reglrise_loss.item(),
                              "total":tot_loss, "best loss":best_loss, "epochs since improv.":num_epochs_without_improvement})
