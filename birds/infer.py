import zuko
import torch
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict
import matplotlib.pyplot as plt

from .plotting import plot_posterior


def _setup_optimizer(model, flow, learning_rate, n_epochs):
    parameters_to_optimize = list(flow.parameters())
    optimizer = torch.optim.AdamW(parameters_to_optimize, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_parameters = sum([len(a) for a in parameters_to_optimize])
    logging.info(f"Training flow with {n_parameters} parameters")
    return optimizer, scheduler


def _setup_loss(loss_name):
    if loss_name == "LogMSELoss":
        loss_fn = torch.nn.MSELoss(reduction="sum")

        def loss(x, y):
            mask = (x > 0) & (y > 0)  # remove points where log is not defined.
            return loss_fn(torch.log10(x[mask]), torch.log10(y[mask]))

    elif loss_name == "MSELoss":
        loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = lambda x, y: loss_fn(x, y)
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
    posteriors_dir.mkdir(exist_ok=True, parents=True)
    return save_dir, posteriors_dir


def _compute_forecast_loss_reverse(model, flow_cond, obs_data, loss_fn, n_samples):
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
        for i in range(len(outputs_list)):
            observation = obs_data[i]
            model_output = outputs_list[i]
            loss_i += loss_fn(observation, model_output)
        loss += loss_i
    return loss / n_samples

def _compute_forecast_loss_forward(model, flow_cond, obs_data, loss_fn, n_samples):
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
        return loss, loss # first is diffed, second is not.

    # Sample from flow
    params_list = flow_cond.rsample((n_samples,))
    # Get Jacobian using Forward-diff
    total_jacobian = torch.zeros(params_list[0].shape)
    # Compute ABM part with Forward-diff
    jacfwd = torch.func.jacfwd(aux_fun, 0, randomness="same", has_aux=True)
    total_loss = 0
    total_params_diff = 0
    for params in params_list:
        jacobian, loss = jacfwd(params.detach()) # Important to detach here since we don't reverse diff this.
        total_loss += loss
        # Use reverse diff for flow
        total_params_diff += torch.dot(jacobian.to(torch.float), params)
    # Back-propagate to flow parameters
    total_params_diff.backward()
    return total_loss


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
    f.savefig(posteriors_dir / f"posterior_{it:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(f)


def infer(
    model: torch.nn.Module,
    flow: zuko.flows.FlowModule,
    prior: torch.distributions.Distribution,
    obs_data: list[torch.Tensor],
    diff_mode = "reverse",
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
    optimizer, scheduler = _setup_optimizer(model, flow, learning_rate, n_epochs)
    loss_fn = _setup_loss(loss)
    save_dir, posteriors_dir = _setup_paths(save_dir)
    w = torch.tensor(w, requires_grad=True)
    iterator = tqdm(range(n_epochs))
    losses = defaultdict(list)
    best_loss = np.inf
    best_forecast_loss = np.inf
    if diff_mode == "reverse":
        forecast_loss_fn = _compute_forecast_loss_reverse
    elif diff_mode == "forward":
        forecast_loss_fn = _compute_forecast_loss_forward
    else:
        raise ValuError(f"Diff mode {diff_mode} not supported.")
    for it in iterator:
        need_to_plot_posterior = plot_posteriors == "every"
        flow_cond = flow(torch.zeros(1, device=device))
        optimizer.zero_grad()
        forecast_loss = forecast_loss_fn(
                model=model,
                flow_cond=flow_cond,
                obs_data=obs_data,
                loss_fn=loss_fn,
                n_samples=n_samples_per_epoch,
            )
        reglrise_loss = w * _get_regularisation(
            flow_cond=flow_cond, prior=prior, n_samples=n_samples_regularization
        )
        loss = forecast_loss + reglrise_loss
        losses["forecast"].append(forecast_loss.item())
        losses["reglrise"].append(reglrise_loss.item())
        losses["total"].append(loss.item())
        if torch.isnan(loss):
            torch.save(flow.state_dict(), save_dir / f"to_debug.pth")
            raise ValueError("Loss is nan!")
        if diff_mode == "reverse":
            loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        if loss.item() < best_loss:
            torch.save(flow.state_dict(), save_dir / f"best_model_{it:04d}.pth")
            best_loss = loss.item()
            need_to_plot_posterior = need_to_plot_posterior or (
                plot_posteriors == "best"
            )
        if forecast_loss.item() < best_forecast_loss:
            torch.save(
                flow.state_dict(), save_dir / f"best_model_forecast_{it:04d}.pth"
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
        scheduler.step()

