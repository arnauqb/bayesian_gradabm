from sbi.inference import prepare_for_sbi, SNPE, SNLE, simulate_for_sbi
import sbi.utils as utils
import torch.nn as nn

def sbi_training(simulator,
                 prior,
                 y,
                 method,
                 density_estimator="maf",
                 n_samples=10_000,
                 n_sims=[10_000], 
                 embedding_net=nn.Identity(),
                 sim_postprocess=lambda x: x,
                 num_workers=1,
                 outloc=None):

    """
    Required inputs:

    - simulator:        callable, consumes torch.tensor containing the input
                        parameters for the model and returns SUMMARY STATISTICS
                        generated by the model. TODO: extend so that we can use
                        embedding networks, although perhapt not possible or
                        necessary for the methods we compare against in this
                        paper.
    - prior:            must have .log_prob and .sample methods. Just use a
                        torch distribution
    - y:                torch.tensor containing observed SUMMARY STATISTICS of 
                        the observed data
    - method:           str, must be in ["SNPE", "SNLE", "SNVI"]

    Optional:
    - density_estimator: can be a string naming one of the density estimators
                         already implemented in sbi, or another custom density
                         estimator
    - n_samples:         int, number of final samples to get from the final 
                         posterior
    - n_sims:            list of ints. Each int in the list specifies number of
                         simulations to generate at each training round.
                         Amortised inference corresponds to list of length 1
    - sim_postprocess:   callable, transforms output of simulator. NOT IMPLEMENTED
    - num_workers:       NOT YET IMPLEMENTED
    - outloc:            str, location to save data. If None, samples aren't saved
    """

    _simulator = simulator
    _ = simulator(prior.sample(1))
    if isinstance(_, list):

        assert len(_) == 1, "Code currently assumes you're simulating a list of length 1"

        def _simulator(theta):
            out = simulator(theta)[0]
            return out

    sbi_simulator, sbi_prior = prepare_for_sbi(_simulator, prior)
    #sbi_simulator = simulator
    #sbi_prior = prior
    posteriors = []
    proposal = sbi_prior

    if method == "SNPE":
        neural_posterior = utils.posterior_nn(
                model="maf", embedding_net=embedding_net, hidden_features=50, num_transforms=5
            )
        inference = SNPE(prior=sbi_prior, density_estimator=neural_posterior)
    elif method in ["SNLE", "SNVI"]:
        inference = SNLE(prior=sbi_prior, density_estimator=density_estimator)

    for sim_count in n_sims:
        #theta = proposal.sample((sim_count,))
        #x = sbi_simulator(theta)
        theta, x = simulate_for_sbi(sbi_simulator, proposal, sim_count, num_workers=num_workers, show_progress_bar=True)
        # This is usually for reshaping for the embedding net. I.e. sbi requires simulator to output
        # single vector, so may need to reshape output of sbi_simulator above
        #x = sim_postprocess(x)

        if method == "SNPE":
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        else:
            _ = inference.append_simulations(theta, x).train()

        if method == "SNPE":
            posterior = inference.build_posterior(density_estimator).set_default_x(y)
        elif method == "SNLE":
            posterior = inference.build_posterior().set_default_x(y)
        elif method == "SNVI":
            posterior = inference.build_posterior(sample_with="vi", vi_method="fKL").set_default_x(y)

        posteriors.append(posterior)
        proposal = posterior

    samples = proposal.sample((n_samples,))

    if not outloc is None:
        torch.save(samples, outloc)

    return proposal, samples
