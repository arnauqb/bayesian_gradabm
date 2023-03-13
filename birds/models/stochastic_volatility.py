import torch
from torch import distributions, nn

"""
This code follows Patrick's misspecification paper example.
"""


class StochVolPrior:
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


class StochVolSimulator(nn.Module):
    def __init__(self, T=100, sigma=0.0):
        super().__init__()
        self._eps = distributions.normal.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        self.T = T
        self.sigma = sigma

    def forward(self, theta):
        """
        Assumes theta is shape (2,), with nu in first entry and tau in second
        """
        nu, tau = theta[0], theta[1]
        nu = torch.max(torch.tensor(0.5, device=nu.device), nu) # To avoid nans.
        tau = torch.max(torch.tensor(10, device=tau.device), tau) # To avoid nans.
        epsilons = self._eps.rsample((self.T + 1,)) / tau
        x = torch.zeros(self.T + 1)
        s = 0.0
        x.requires_grad_ = True
        for t in range(self.T + 1):
            s = s + epsilons[t]
            obs_density = distributions.studentT.StudentT(
                nu, torch.tensor([0.0]), torch.exp(s)
            )
            obs = obs_density.rsample()
            factor = 1.0
            if (t >= 50) and (t <= 65):
                factor += 5.0 * self.sigma
            x[t] = obs * factor
        return [x]
