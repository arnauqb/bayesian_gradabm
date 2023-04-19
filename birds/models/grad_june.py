import torch
from grad_june import Runner


class BirdsJUNE(torch.nn.Module):
    def __init__(self, runner, params_to_calibrate, data_to_calibrate):
        super().__init__()
        self.runner = runner
        self.params_to_calibrate = params_to_calibrate
        self.data_to_calibrate = data_to_calibrate
        self.device = runner.device

    @classmethod
    def from_config(cls, config, parameters_to_calibrate, data_to_calibrate):
        runner = Runner.from_parameters(config)
        return cls(runner, parameters_to_calibrate, data_to_calibrate)

    @classmethod
    def from_file(cls, file_path, parameters_to_calibrate, data_to_calibrate):
        runner = Runner.from_file(file_path)
        return cls(runner, parameters_to_calibrate, data_to_calibrate)

    def forward(self, params):
        for (j, name) in enumerate(self.params_to_calibrate):
            if name == "seed":
                self.runner.log_fraction_initial_cases = torch.min(
                    torch.tensor(-1.0), params[j]
                )
            elif name.startswith("beta"):
                beta_name = "_".join(name.split("_")[1:])
                self.runner.model.infection_networks.networks[beta_name].log_beta = params[j]
            elif name.startswith("sd"):
                sd_name = "_".join(name.split("_")[1:])
                factor = torch.sigmoid(params[j]) # guarantees between 0 and 1
                self.runner.model.policies.interaction_policies[0].beta_factors[sd_name] = factor
            else:
                raise ValueError(f"Parameter name {name} not recognized.")
        res, _ = self.runner()
        ret = []
        for key in self.data_to_calibrate:
            if "age" in key:
                age_bin = int(key.split("_")[-1])
                toappend = res[key] 
            elif key == "cases_per_timestep" or key == "deaths_per_timestep":
                toappend = res[key]
            else:
                raise ValueError(f"Data to calibrate {key} not supported.")
            ret.append(toappend)
        return ret
