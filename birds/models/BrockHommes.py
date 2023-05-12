import torch
import torch.distributions as distributions

sftmx = nn.Softmax(dim=-1)

class BrockHommes(nn.Module):

    def __init__(self, T=100):

        super().__init__()
        self._T = T
        self.device = 'cpu'
        self._eps = distributions.normal.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        )

    def forward(self, theta):

        """
        theta is a torch.tensor of shape (theta_dim,) which looks like

        beta, g1, g2, g3, g4, b1, b2, b3, b4, sigma, r

        See Equations 39 and 40 of https://arxiv.org/pdf/2202.00625.pdf for reference
        """

        beta = theta[0]
        assert beta > 0, "Intensity of choice should be > 0"
        g = theta[1:5]
        b = theta[5:9]
        sigma = theta[-2]
        assert sigma > 0, "Noise level should be > 0"
        r = theta[-1]
        assert r >= 0, "Prevailing interest rate should be >= 0"
        R = 1. + r

        epsilons = self._eps.rsample((self._T,))
        x = torch.zeros(self._T)
        x.requires_grad_ = True
        for t in range(self._T):
            exponent = beta * (x[t-1] - R * x[t-2]) * (g * x[t-3] + b - R * x[t-2])
            norm_exponentiated = sftmx(exponents)
            mean = (norm_exponentiated * (g * x[t-1] + b)).sum()
            x[t] = (mean + epsilons[t] * sigma) / R
        return x
