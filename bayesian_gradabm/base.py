from abc import ABC
import pandas as pd
import yaml
import pickle
import sys
import torch
from pathlib import Path
import torch.distributions as dist

from grad_june import Runner
from .utils import get_attribute, set_attribute, read_device


class InferenceEngine(ABC):
    def __init__(
        self,
        runner,
        priors,
        observed_data,
        data_observable,
        training_configuration,
        results_path,
        device,
    ):
        super().__init__()
        self.runner = runner
        self.priors = priors
        self.observed_data = observed_data
        self.data_observable = data_observable
        self.training_configuration = training_configuration
        self.results_path = self._read_path(results_path)
        self.device = device

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, parameters):
        with open(parameters["june_configuration_file"], "r") as f:
            june_params = yaml.safe_load(f)
        device = parameters["device"]
        june_params["system"]["device"] = device
        runner = Runner.from_parameters(june_params)
        priors = cls.read_parameters_to_fit(parameters)
        observed_data = cls.load_observed_data(parameters, device)
        data_observable = parameters["data"]["observable"]
        training_configuration = parameters.get("training", {})
        return cls(
            runner=runner,
            priors=priors,
            results_path=parameters["results_path"],
            observed_data=observed_data,
            data_observable=data_observable,
            device=device,
            training_configuration=training_configuration,
        )

    @classmethod
    def read_parameters_to_fit(cls, params):
        parameters_to_fit = params["parameters_to_fit"]
        ret = {}
        for key in parameters_to_fit:
            dist_info = parameters_to_fit[key]["prior"]
            dist_class = getattr(dist, dist_info.pop("dist"))
            dist_info = {
                key: torch.tensor(dist_info[key], device=params["device"])
                for key in dist_info
            }
            ret[key] = dist_class(**dist_info)
        return ret

    @classmethod
    def load_observed_data(cls, params, device):
        data_params = params["data"]
        df = pd.read_csv(data_params["observed_data"], index_col=0)
        ret = {}
        for key in df:
            ret[key] = torch.tensor(df[key], device=device, dtype=torch.float)
        return ret

    def _set_initial_parameters(self):
        names_to_save = []
        with torch.no_grad():
            for param_name in self.priors:
                if param_name == "model.infection_networks.networks.leisure.log_beta":
                    for _name in ["pub", "grocery", "gym", "cinema", "visit"]:
                        name = param_name.split(".")
                        name[3] = _name
                        name = ".".join(name)
                        set_attribute(self.runner, name, self.priors[param_name].mean())
                else:
                    set_attribute(self.runner, param_name, self.priors[param_name].mean())
                names_to_save.append(param_name)
        return names_to_save

    def _read_path(self, results_path):
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)
        return results_path

    def evaluate(self, samples):
        raise NotImplementedError

    def save_results(self, path):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
