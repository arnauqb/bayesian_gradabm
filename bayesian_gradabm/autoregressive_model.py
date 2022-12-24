import torch
import pyro


def run_autoregressive_model(parameters, n_timesteps=50, sigma_noise=1):
    """
    Runs an autoregressive model ( https://en.wikipedia.org/wiki/Autoregressive_model ).
    For a set of parameters over two batch dimensions.

    Parameters
    ----------
    parameters:
        Tensor of parameters of shape [batch_dim_1, batch_dim_2, parameters_dim]
    n_timesteps:
        Number of time steps to simulate
    sigma_noise:
        std of the Gaussian noise.
    """
    assert len(parameters.shape) == 3
    batch_dim_1 = parameters.shape[0]
    batch_dim_2 = parameters.shape[1]
    theta_dim = parameters.shape[-1]
    # AR(0)
    epsilon_dist = torch.distributions.Normal(
        torch.zeros(1), sigma_noise * torch.ones(1)
    )
    # Want to batch simulations, so generate epsilons for each simulation
    # (of which there are n_x_conds * n_params_per_x_cond)
    epsilon = epsilon_dist.sample(
        (
            batch_dim_1,
            batch_dim_2,
        )
    )[:, :, 0]
    theta_0s = torch.clone(parameters[:, :, 0])
    x = (theta_0s + epsilon).unsqueeze(-1)
    for i in range(1, n_timesteps):
        x_i = torch.clone(parameters[:, :, 0])
        for p in range(1, min(i, theta_dim)):
            added_term = parameters[:, :, p - 1] * x[:, :, i - p]
            x_i += added_term
        epsilon_i = epsilon_dist.sample(
            (
                batch_dim_1,
                batch_dim_2,
            )
        )[:, :, 0]
        x_i = x_i + epsilon_i
        x_i = x_i.unsqueeze(-1)
        x = torch.cat((x, x_i), dim=-1)
    return x.unsqueeze(-1)


def autoregressive_model_pyro(
    data_obs, n_parameters, n_batch, n_timesteps=50, sigma_noise=1
):
    """
    Pyro model for the autoregressive model. Only supports up to 2 parameters.

    Parameters
    ----------
    data_obs:
        Observed time series to fit to.
    n_parameters:
        Number of parameters, 1, or 2.
    n_batch:
        batch number. Set to 1 for now.
    n_timesteps:
        Number of time steps to simulate
    sigma_noise:
        std of the Gaussian noise.
    """
    prior_phi_1 = pyro.distributions.Uniform(-1, 1).expand((n_batch, 1))
    phi_1 = pyro.sample("phi_1", prior_phi_1)
    if n_parameters == 1:
        phi = phi_1.reshape(1, -1)
    else:
        prior_phi_2 = pyro.distributions.Uniform(
            -1, torch.min(1 - phi_1, 1 + phi_1)
        ).expand((n_batch, 1))
        phi_2 = pyro.sample("phi_2", prior_phi_2)
        phi = torch.hstack((phi_1, phi_2))
    # AR(0)
    # first point
    x = torch.clone(phi[:, 0])
    epsilon_dist_0 = pyro.distributions.Normal(x, sigma_noise * torch.ones(n_batch))
    x = pyro.sample(f"x_{0}", epsilon_dist_0, obs=data_obs[0]).reshape(1, -1)
    for i in range(1, n_timesteps):
        x_i = torch.clone(phi[:, 0])
        for p in range(1, min(i, n_parameters)):
            x_j = x[:, i - p]
            x_i += x_j * phi[:, p - 1]
        epsilon_dist_i = pyro.distributions.Normal(
            x_i, sigma_noise * torch.ones(n_batch)
        )
        x_i = pyro.sample(f"x_{i}", epsilon_dist_i, obs=data_obs[i])
        x = torch.hstack((x, x_i.reshape(1, -1)))
    return x


def fit_autoregressive_model_pyro(
    data_obs, sigma_noise, num_samples=1000, warmup_steps=1000, n_parameters=1
):
    """
    Fits an autoregressive model using Pyro.

    Parameters
    ----------
    data_obs:
        Observed time series to fit to.
    sigma_noise:
        std of the Gaussian noise.
    num_samples:
        Number of chain samples.
    warmup_steps:
        Number of warmup steps.
    n_parameters:
        Number of parameters, 1, or 2.
    """
    n_timesteps = data_obs.shape[0]
    kernel = pyro.infer.mcmc.NUTS(autoregressive_model_pyro)
    mcmc = pyro.infer.mcmc.MCMC(
        kernel, num_samples=num_samples, warmup_steps=warmup_steps
    )
    mcmc.run(
        data_obs,
        n_parameters=n_parameters,
        n_batch=1,
        n_timesteps=n_timesteps,
        sigma_noise=sigma_noise,
    )
    return mcmc
