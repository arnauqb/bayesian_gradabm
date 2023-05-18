import torch
import torch.nn as nn


class FrankeWesterhoff(nn.Module):
    def __init__(self, flavor='hpm'):
        super().__init__()
        
        self.flavour = flavour
        self.mu = torch.tensor(0.01)
        self.beta = torch.tensor(1)
        
        if self.flavour == "hpm":
            self.phi = torch.tensor(0.12)
            self.chi = torch.tensor(1.5)
            self.sigmaf = torch.tensor(0.758)
        elif self.flavour == "wp":
            self.phi = torch.tensor(1)
            self.chi = torch.tensor(0.9)
            self.alpha0 = torch.tensor(2.1)
            self.sigmaf = torch.tensor(0.752)
            self.scale = torch.tensor([15000., 1., 5.])
            
    def _hpm(self, mu, beta, phi, chi, alpha0, alphan, alphap, sigmaf, sigmac, T, seed):
        if not (seed is None):
            torch.random.seed(seed)
        
        def _hpm_mean(self, mu, nf, phi, chi, pt1, pt2):
            return ( ( 1 - mu * ( nf * phi - ( 1 - nf ) * chi) ) * pt1 - ( mu * (1 - nf) * chi ) * pt2 )

        def _hpm_std(self, mu, nf, sigmaf, sigmac):
            return mu * np.sqrt( ( nf * sigmaf ) ** 2 + ( (1 - nf) * sigmac ) ** 2 )

        def _hpm_a(self, alphan, nf, alpha0, alphap, p):
            return alphan * ( 2*nf - 1 ) + alpha0 + alphap * (p**2)

        p = torch.zeros_like(T+1)
        a = alpha0
        es = torch.randn(T)

        for t in range(1, T):
            nf = 1. / (1. + torch.exp( - beta * a ))
            p = torch.clone(p)
            
            p[t] = self._hpm_mean(mu, nf, phi, chi, p[t-1], p[t-2]) + self._hpm_std(mu, nf, sigmaf, sigmac) * es[t-1]
            a = self._hpm_a(alphan, nf, alpha0, alphap, p[t-1])
                
        return p[1:-1]
    
    def _wp(self, mu, beta, phi, chi, alpha0, alphaw, eta, p_star, esf, esc, wf, wc, a, p, df, dc, seed):
        
        if not (seed is None):
            torch.random.seed(seed)

        gf = 0.
        gc = 0.
        nf = 0.5

        for t in range(2, p.shape[0]-1):

            gf = (torch.exp(p[t]) - torch.exp(p[t - 1])) * df[t - 2]
            gc = (torch.exp(p[t]) - torch.exp(p[t - 1])) * dc[t - 2]

            wf = torch.clone(wf)
            wc = torch.clone(wc)
            a = torch.clone(a)
            dc = torch.clone(dc)
            df = torch.clone(df)
            
            # Wealth updates
            wf_new[t] = eta * wf_old[t - 1] + (1 - eta) * gf
            wc_new[t] = eta * wc_old[t - 1] + (1 - eta) * gc

            a[t] = alphaw * (wf[t] - wc[t]) + alpha0

            # New proportion of agents following strategy f
            nf = 1 / (1 + np.exp(-beta * a[t - 1]))

            # Demand updates
            dc[t] = chi * (p[t] - p[t - 1]) + esc[t] 
            df[t] = phi * (p_star - p[t]) + esf[t]

            p = torch.clone(p)
            # Price update
            p[t + 1] = p[t] + mu * (nf * df[t] + (1 - nf) * dc[t])	

        # We're calibrating against the time-series of log-returns
        x = p[1:]
        x_diff = x[1:] - x[:-1] # implements np.diff(x)
        return x_diff[1:] #np.diff(p[1:])[1:]
    
    
    def forward(self, pars, T=50, seed=None):
        
        if self.flavor == "hpm":
            alpha0, alphan, alphap, sigmac = [float(pars[i]) for i in range(4)]
            return self._hpm(self.mu, self.beta, self.phi, self.chi, alpha0, alphan, alphap, self.sigmaf, sigmac, T, seed)
        
        elif self.flavor == "wp":
            alphaw, eta, sigmac = [float(pars[i])*self.scale[i] for i in range(3)]
            
            esf, esc = self.sigmaf * torch.randn(size=T), sigmac * torch.randn(size=T)
            wf = torch.zeros_like(T)
            wc = torch.zeros_like(T)
            a = torch.zeros_like(T)
            p = torch.zeros_like(T + 1)
            df = torch.zeros_like(T)
            dc = torch.zeros_like(T)
            return _wp(self.mu, self.beta, self.phi, self.chi, self.alpha0, alphaw, eta, 0., esf, esc, wf, wc, a, p, df, dc, seed)