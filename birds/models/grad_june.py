import torch
from grad_june import Runner


class BirdsJUNE(torch.nn.Module):
    def __init__(self, runner, params_to_calibrate, data_to_calibrate):
        super().__init__()
        self.runner = runner
        self.params_to_calibrate = params_to_calibrate
        self.data_to_calibrate = data_to_calibrate

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
            self.runner.model.infection_networks.networks[name].log_beta = params[j]
        res, _ = self.runner()
        res["daily_deaths"] = self.runner.data.results["daily_deaths"]
        ret = []
        for key in self.data_to_calibrate:
            ret.append(res[key])
        return ret
