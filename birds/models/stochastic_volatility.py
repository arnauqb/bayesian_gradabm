import torch
from torch import distributions, nn

"""
This code follows Patrick's misspecification paper example.
"""


class _StochVolPrior:
    def __init__(self):

        self._nu = distributions.gamma.Gamma(5, 1)
        self._tau = distributions.gamma.Gamma(5, 1.0 / 25.0)

    def rsample(self, n_samples):

        nu_samples = self._nu.rsample((n_samples, 1))
        tau_samples = self._tau.rsample((n_samples, 1))
        return torch.cat((nu_samples, tau_samples), dim=-1)

    def log_prob(self, samples):

        """
        Assumes samples are of shape (n_samples, 2), where first column is
        nu samples and second is tau samples.

        Returns the log prob of each row, so would need to sum to get total
        log prob
        """

        nus, taus = samples[:, 0], samples[:, 1]
        nus_log_prob = self._nu.log_prob(nus)
        taus_log_prob = self._tau.log_prob(taus)
        return nus_log_prob + taus_log_prob

class StochVolPrior(distributions.distribution.Distribution):
    def __init__(self):

        self._log_nu = distributions.normal.Normal(1.5, 0.25)
        self._log_tau = distributions.normal.Normal(-2, 0.5)

    def rsample(self, n_samples):

        log_nu_samples = self._log_nu.rsample((n_samples, 1))
        log_tau_samples = self._log_tau.rsample((n_samples, 1))
        return torch.cat((log_nu_samples, log_tau_samples), dim=-1)

    def sample(self, n_samples):

        if isinstance(n_samples, tuple):
            if len(n_samples) > 0:
                n_samples = n_samples[0]
            elif len(n_samples) == 0:
                n_samples = 1
        log_nu_samples = self._log_nu.sample((n_samples, 1))
        log_tau_samples = self._log_tau.sample((n_samples, 1))
        return torch.cat((log_nu_samples, log_tau_samples), dim=-1)

    def log_prob(self, samples):

        """
        Assumes samples are of shape (n_samples, 2), where first column is
        nu samples and second is tau samples.

        Returns the log prob of each row, so would need to sum to get total
        log prob
        """

        log_nus, log_taus = samples[..., 0], samples[..., 1]
        log_nus_log_prob = self._log_nu.log_prob(log_nus)
        log_taus_log_prob = self._log_tau.log_prob(log_taus)
        return log_nus_log_prob + log_taus_log_prob


class StochVolSimulator(nn.Module):
    def __init__(self, T=100, sigma=0.0):
        super().__init__()
        self._eps = distributions.normal.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        self.T = T
        self.sigma = sigma
        self.device = "cpu"

    def forward(self, log_theta):
        """
        Assumes theta is shape (2,), with nu in first entry and tau in second
        """
        theta = torch.exp(log_theta)
        nu, tau = theta[..., 0] + 1, theta[..., 1] + 1
        epsilons = self._eps.rsample((self.T + 1,)) / tau
        x = torch.zeros(self.T + 1)
        s = 0.0
        x.requires_grad_ = True
        mask = torch.zeros(self.T + 1, dtype=int)
        # to introduce miss-specification
        mask[50:65] = 1
        for t in range(self.T + 1):
            s = s + epsilons[t]
            obs_density = distributions.studentT.StudentT(
                nu, torch.tensor([0.0]), torch.exp(s)
            )
            obs = obs_density.rsample()
            factor = 1.0
            factor = factor + mask[t] * 5 * self.sigma
            x[t] = obs * factor
        return [x]
