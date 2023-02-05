import torch
import normflows as nf
from tqdm import tqdm
import yaml
import corner
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse

from grad_june import Runner

parser = argparse.ArgumentParser(prog="Fit June with flows")
parser.add_argument("-K", "--ntransforms")
parser.add_argument("-L", "--lr")
parser.add_argument("-B", "--batch")
parser.add_argument("-S", "--hidden_size")
parser.add_argument("-d", "--device")
args = parser.parse_args()

device = str(args.device)

true_parameters = {}
# true_parameters = {
#  "care_home":
#    {"log_beta": 1.3054601},
#  "care_visit":
#    {"log_beta": 0.46661767},
#  "cinema":
#    {"log_beta": 0.7523964},
#  "company":
#    {"log_beta": 0.3428555},
#  "grocery":
#    {"log_beta": 0.7523964},
#  "gym":
#    {"log_beta": 0.7523964},
#  "household":
#    {"log_beta": 0.30810425},
#  "pub":
#    {"log_beta": 0.7523964},
#  "school":
#    {"log_beta": 0.4572914},
#  "university":
#    {"log_beta": 0.34417054},
#  "visit":
#    {"log_beta": 0.7523964},
# }
true_parameters["household"] = {"log_beta": 1.0}
true_parameters["company"] = {"log_beta": 1.0}
true_parameters["school"] = {"log_beta": 1.0}
# true_parameters["pub"] = {"log_beta" : 1.0}
# true_parameters["visit"] = {"log_beta" : 1.0}

n_true_parameters = len(true_parameters)
true_values = np.array([true_parameters[key]["log_beta"] for key in true_parameters])
param_names = list(true_parameters.keys())

params = yaml.safe_load(open("./configs/best_run.yaml"))
params["timer"]["total_days"] = 10
params["system"]["device"] = device
# params["data_path"] = "/cosma7/data/dp004/dc-quer1/torch_june_worlds/data_camden.pkl"
params["data_path"] = "/Users/arnull/code/gradabm-june/test/data/data.pkl"


for key in true_parameters:
    params["networks"][key] = true_parameters[key]
runner = Runner.from_parameters(params)

true_data = runner()[0]["cases_per_timestep"]
n_people = runner.n_agents
true_data = true_data / n_people


prior = torch.distributions.Normal(
    torch.zeros(n_true_parameters, device=device),
    torch.ones(n_true_parameters, device=device),
)

# Set up model

# Define flows
def setup_flow(K, hidden_size, device):
    latent_size = len(true_values)
    hidden_units = hidden_size
    hidden_layers = 2
    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]
    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(len(true_values), trainable=False)
    # Construct flow model
    flow = nf.NormalizingFlow(q0=q0, flows=flows)
    # Move model on GPU if available
    flow = flow.to(device)
    return flow


def run_model(sample):
    # print(sample)
    sample = sample.flatten()
    for (j, name) in enumerate(true_parameters):
        runner.model.infection_networks.networks[name].log_beta = sample[j]
    cases_per_timestep = runner()[0]["cases_per_timestep"] / n_people
    return cases_per_timestep


def get_forecast_score(flow, true_data, loss_fn, n_samples=5):
    loss = 0.0
    for i in range(n_samples):
        sample, lp = flow.sample()
        # print(sample)
        cases_per_timestep = run_model(sample)
        # print(cases_per_timestep)
        loss += loss_fn(cases_per_timestep, true_data)
        # print(loss)
    return loss / n_samples


def get_regularisation(flow, n_samples=5):
    samples, flow_lps = flow.sample(n_samples)
    prior_lps = prior.log_prob(samples).sum(1)
    kl = torch.mean(flow_lps - prior_lps)
    return kl


def train(
    ntransforms,
    lr,
    batch_size,
    hidden_size,
    device,
):
    flow = setup_flow(K=ntransforms, hidden_size=hidden_size, device=device)
    losses = defaultdict(list)
    best_loss = np.inf
    # Train model

    parameters_to_optimize = list(flow.parameters())
    print(sum([len(a) for a in parameters_to_optimize]))
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    n_epochs = 1000
    n_samples_per_epoch = batch_size
    n_samples_reg = 10

    w = 0.0  # torch.tensor(1.0, requires_grad=True)

    iterator = tqdm(range(n_epochs))

    for it in iterator:
        optimizer.zero_grad()
        forecast_loss = get_forecast_score(
            flow=flow,
            true_data=true_data,
            loss_fn=loss_fn,
            n_samples=n_samples_per_epoch,
        )
        reglrise_loss = get_regularisation(flow=flow, n_samples=n_samples_reg)
        loss = forecast_loss + w * reglrise_loss
        losses["forecast_train"].append(forecast_loss.item())
        losses["reglrise_train"].append(reglrise_loss.item())
        # print(loss)
        if torch.isnan(loss):
            print("loss is nan!")
            break
        loss.backward()

        optimizer.step()
        name = f"model_{ntransforms}_{batch_size}_{hidden_size}_{lr}"

        with torch.no_grad():
            val_forecast_loss = get_forecast_score(
                flow=flow,
                true_data=true_data,
                loss_fn=loss_fn,
                n_samples=n_samples_per_epoch,
            )
            val_reglrise_loss = get_regularisation(flow=flow, n_samples=n_samples_reg)
            val_loss = val_forecast_loss + w * val_reglrise_loss

            losses["forecast_val"].append(val_forecast_loss.item())
            losses["reglrise_val"].append(val_reglrise_loss.item())

            if val_loss.item() < best_loss:
                torch.save(flow.state_dict(), name + ".pth")
                best_loss = val_loss.item()
            iterator.set_postfix(
                {
                    "fl": forecast_loss.item(),
                    "rl": reglrise_loss.item(),
                    "val loss": val_loss.item(),
                    "best val loss": best_loss,
                }
            )
        df = pd.DataFrame(losses)
        df.to_csv("losses_" + name + ".csv")


train(
    ntransforms=int(args.ntransforms),
    lr=float(args.lr),
    batch_size=int(args.batch),
    hidden_size=int(args.hidden_size),
    device=str(args.device),
)
