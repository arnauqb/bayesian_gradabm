import sys
from bayesian_gradabm.normflows import NormFlows
from copy import deepcopy
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index")
parser.add_argument("-f", "--file")
args = parser.parse_args()
i = int(args.index)
f = str(args.file)

n_devices = 10
cuda_device = i % n_devices
gpu = True


lr = 1e-4
batch = 10
hidden_size_range = [16, 32, 64, 128]
k_range = [2, 4, 8, 16]

base_params = yaml.safe_load(open(f))

j = 0
for lr in lr_range:
    for batch in batch_range:
        for hidden_size in hidden_size_range:
            for k in k_range:
                if i != j:
                    j += 1
                    continue
                name = f"/{lr}_{batch}_{hidden_size}_{k}"
                params = deepcopy(base_params)
                if gpu:
                    params["device"] = f"cuda:{cuda_device}"
                else:
                    params["device"] = "cpu"
                params["results_path"] += name
                params["training"]["n_batch"] = batch
                params["training"]["optimizer"]["lr"] = lr
                params["training"]["flow"]["K"] = k
                params["training"]["flow"]["AutoregressiveRationalQuadraticSpline"][
                    "num_hidden_channels"
                ] = hidden_size
                nf = NormFlows.from_parameters(params)
                nf.run()

