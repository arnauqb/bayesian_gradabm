from pathlib import Path
import pymultinest
from pymultinest.solve import solve
import yaml
import torch
import numpy as np
import pandas as pd
from scipy import stats

from grad_june import GradJune
from .utils import read_fortran_data_file, set_attribute
from .base import InferenceEngine


def read_pyro_to_scipy(dist, **kwargs):
    if dist == "Uniform":
        return stats.uniform(loc=kwargs["low"], scale=kwargs["high"] - kwargs["low"])
    elif dist == "Normal":
        return stats.norm(loc=kwargs["loc"], scale=kwargs["scale"])
    else:
        raise NotImplementedError


class MultiNest(InferenceEngine):
    @classmethod
    def read_parameters_to_fit(cls, params):
        parameters_to_fit = params["parameters_to_fit"]
        ret = {}
        for key in parameters_to_fit:
            ret[key] = read_pyro_to_scipy(**parameters_to_fit[key]["prior"])
        return ret

    def _prior(self, cube):
        """
        TODO: Need to invert from unit cube for other distros.
        """
        params = cube.copy()
        for i, key in enumerate(self.priors):
            params[i] = self.priors[key].ppf(cube[i])
        return params

    def _loglike(self, cube):
        # Set model parameters
        likelihood_fn = getattr(
            torch.distributions, self.training_configuration["likelihood"]
        )
        with torch.no_grad():
            # self.runner.reset_model()
            samples = {}
            for i, key in enumerate(self.priors):
                samples[key] = torch.tensor(cube[i], device=self.device)
            y = self.evaluate(samples)
            # Compare to data
            ret = 0.0
            for key in self.data_observable:
                time_stamps = self.data_observable[key]["time_stamps"]
                if time_stamps == "all":
                    time_stamps = range(len(y[key]))
                data = y[key][time_stamps]
                rel_error = self.data_observable[key]["error"]  * data
                data_obs = self.observed_data[key][time_stamps]
                error = rel_error + 10
                l_ = likelihood_fn(data, error).log_prob(data_obs)
                ret += l_.sum().cpu().item()
            return ret

    def evaluate(self, samples):
        with torch.no_grad():
            for param_name in samples:
                if param_name == "model.infection_networks.networks.leisure.log_beta":
                    for _name in ["pub", "grocery", "gym", "cinema", "visit"]:
                        name = param_name.split(".")
                        name[3] = _name
                        name = ".".join(name)
                        set_attribute(self.runner, name, samples[param_name])
                else:
                    set_attribute(self.runner, param_name, samples[param_name])
        results, _ = self.runner()
        return results

    def run(self, **kwargs):
        self._set_initial_parameters()
        ndims = len(self.priors)
        result = solve(
            LogLikelihood=self._loglike,
            Prior=self._prior,
            n_dims=ndims,
            outputfiles_basename=(self.results_path / "multinest").as_posix(),
            verbose=True,
            n_iter_before_update=1,
            resume=False,
            **kwargs,
        )
        self.results = self.save_results()

    def save_results(self):
        results = read_fortran_data_file(self.results_path / "multinest.txt")
        df = pd.DataFrame()
        df["likelihood"] = results[:, 1]
        for i, name in enumerate(self.priors):
            df[name] = results[:, 2 + i]
        df["weights"] = results[:, 0]
        df.to_csv(self.results_path / "results.csv")
        return df
