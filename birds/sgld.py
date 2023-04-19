import torch
import math
import numpy as np

class SGLD(torch.optim.Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 noise_scaling=1.,
                 betas=(0.9, 0.999),
                 num_pseudo_batches=1,
                 num_burn_in_steps=30,
                 eps=1e-8,
                 weight_decay=0,
                 use_barriers=False) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        noise_scaling : float, optional
            Scaling factor used for the noise that is added after the burn-in phase.
            Default: `10`.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `30`.
        eps : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        """
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            noise_scaling=noise_scaling,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            eps=eps,
            betas=betas,
            use_barriers=use_barriers,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group["eps"]
            noise_scaling = group["noise_scaling"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                gradient = p.grad.data

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(p)
                    if group["use_barriers"]:
                        if np.isinf(group["limit_min"]) or np.isinf(group["limit_max"]):
                            state["barrier_coeff"] = 0.
                        else:
                            state["barrier_coeff"] = 0.01 * (group["limit_max"] - group["limit_min"])
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["iteration"] += 1

                grad = p.grad
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state["iteration"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # this is the preconditioner in SGLD
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)                

                if group["use_barriers"]:
                    p.data.sub_(-state["barrier_coeff"] * lr/(p-group["limit_min"]) + state["barrier_coeff"] * lr/(group["limit_max"] - p))

                if state["iteration"] > group["num_burn_in_steps"]:
                    noise = (
                        torch.normal(
                            mean=torch.zeros_like(gradient),
                            std=torch.ones_like(gradient)
                        ) * num_pseudo_batches * noise_scaling
                    )
                    # XXX use sqrt(step_size) to scale the noise properly
                    # This is the difference compared to sgld_original.py
                    p.addcdiv_(exp_avg * noise, denom, value=math.sqrt(step_size))

        return loss
