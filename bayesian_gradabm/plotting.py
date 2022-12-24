import corner

def plot_flow_posterior(flow, x_enc, true_parameters):
    flow_c = flow(x_enc)
    samples = flow_c.sample((10000,))
    samples = samples.cpu().numpy()
    samples = samples.reshape(-1, len(true_parameters))
    f = corner.corner(
        samples,
        labels=[f"p_{i}" for i in range(len(true_parameters))],
        smooth=2,
        truths=true_parameters,
        show_titles=True,
    )
    return
