import torch
import corner

def plot_posterior(flow_cond, param_names=None, true_values=None, lims=(0, 2)):
    with torch.no_grad():
        samples  = flow_cond.sample((10000,))
        samples = samples.cpu().numpy()
        f = corner.corner(
            samples,
            labels=param_names,
            smooth=2,
            show_titles=True,
            bins=30,
            truths=true_values,
            range=lims,
        )
    return f
