import ultranest
import torch
import pickle
from pathlib import Path
from scipy import stats

from .base import InferenceEngine


def read_pyro_to_scipy(dist, **kwargs):
    if dist == "Uniform":
        return stats.uniform(loc=kwargs["low"], scale=kwargs["high"] - kwargs["low"])
    elif dist == "Normal":
        return stats.norm(loc=kwargs["loc"], scale=kwargs["scale"])
    else:
        raise NotImplementedError


class UltraNest(InferenceEngine):
    @classmethod
    def read_parameters_to_fit(cls, params):
        parameters_to_fit = params["parameters_to_fit"]
        ret = {}
        for key in parameters_to_fit:
            ret[key] = read_pyro_to_scipy(**parameters_to_fit[key]["prior"])
        return ret

    def prior_transform(self, cube):
        """ """
        params = cube.copy()
        for i, key in enumerate(self.priors):
            params[i] = self.priors[key].ppf(cube[i])
        return params

    def likelihood(self, params):
        # Set model parameters
        likelihood_fn = getattr(
            torch.distributions, self.training_configuration["likelihood"]
        )
        with torch.no_grad():
            samples = {}
            for i, key in enumerate(self.priors):
                samples[key] = torch.tensor(params[i], device=self.device)
            y = self.evaluate(samples)
            # Compare to data
            ret = 0.0
            for key in self.data_observable:
                time_stamps = self.data_observable[key]["time_stamps"]
                if time_stamps == "all":
                    time_stamps = range(len(y[key]))
                # data = y[key]
                data = y[key][time_stamps]
                data_obs = self.observed_data[key][time_stamps]
                rel_error = self.data_observable[key]["error"]
                data_obs = data_obs[data > 0]
                data = data[data > 0]
                ret += (
                    likelihood_fn(data, rel_error * data)
                    .log_prob(data_obs)
                    .sum()
                    .cpu()
                    .item()
                )
            return ret

    def run(self, **kwargs):
        self._set_initial_parameters()
        param_names = list(self.priors.keys())
        sampler = ultranest.ReactiveNestedSampler(
            param_names,
            self.likelihood,
            self.prior_transform,
            log_dir=self.results_path.as_posix(),
        )
        results = sampler.run(max_ncalls=100)
        sampler.print_results()
        with open(Path(sampler.logs["run_dir"]) / "results.pkl", "wb") as f:
            pickle.dump(results, f)
