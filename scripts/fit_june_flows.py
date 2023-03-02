import torch
from tqdm import tqdm
import yaml
import corner
import random
import zuko
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from grad_june import Runner

params_to_calibrate = ["household", "school"] #, "company"]
device = "cuda:0"
n_epochs = 500
#data_path = "/Users/arnull/code/gradabm-june/worlds/data_camden.pkl"
data_path = "/home/coml0859/data_camden.pkl"

def plot_posterior(nfm, param_names, truths=None, lims=(0, 2)):
    with torch.no_grad():
        samples,_  = nfm.sample(10000)
        samples = samples - 2.0
        samples = samples.cpu().numpy()
        f = corner.corner(
            samples,
            labels=param_names,
            smooth=2,
            show_titles=True,
            bins=30,
            truths=truths,
            range=[lims for i in range(len(params_to_calibrate))],
        )
    return f


def init_runner():
    params = yaml.safe_load(open("/home/coml0859/bayesian_gradabm/configs/tests.yaml"))
    params["networks"]["household"]["log_beta"] = 0.5
    params["networks"]["school"]["log_beta"] = 0.5
    #params["networks"]["company"]["log_beta"] = 0.5
    true_values = [params["networks"][name]["log_beta"] for name in params_to_calibrate]
    n_params = len(params_to_calibrate)
    params["timer"]["total_days"] = 25
    params["timer"]["initial_day"] = "2020-03-01"
    params["system"]["device"] = device
    params["data_path"] = data_path
    runner = Runner.from_parameters(params)
    return runner


def setup_flow():
    return flow

def run_model(runner, sample):
    sample = sample.flatten()
    for (j, name) in enumerate(params_to_calibrate):
        value_disp = sample[j] - 2.0
        if name == "log_fraction_initial_cases":
            runner.log_fraction_initial_cases = torch.minimum(
                torch.tensor(0.0), value_disp
            )
        elif name == "leisure":
            for _name in ["pub", "grocery", "gym", "cinema", "visit"]:
                runner.model.infection_networks.networks[_name].log_beta = value_disp
        else:
            runner.model.infection_networks.networks[name].log_beta = value_disp
    res, _ = runner()
    return res


def get_forecast_score(runner, flow, true_res, loss_fn, n_samples=5):
    loss = 0.0
    for i in range(n_samples):
        sample, lp = flow.sample()
        print(sample)
        print(lp)
        res = run_model(runner, sample)
        #loss_i = 0.0
        #for key in res:
        #    j = 0
        #    if "age" in key:
        #        loss_i += loss_fn(
        #                true_res[key] / runner.population_by_age[j],
        #                res[key] / runner.population_by_age[j],
        #        )
        #        j += 1
        loss_i = loss_fn(
                true_res["cases_per_timestep"]/ runner.n_agents,
                res["cases_per_timestep"]/ runner.n_agents,
        )
        loss_i.backward()
        loss += loss_i
    return loss / n_samples


def get_regularisation(flow, prior, n_samples=5):
    samples, flow_lps = flow.sample(n_samples)
    prior_lps = prior.log_prob(samples).sum(1)
    kl = torch.mean(flow_lps - prior_lps)
    return kl

def train(weight):
    # Train model
    n_epochs = 1000
    runner = init_runner()
    true_res, _ = runner()
    flow = setup_flow()
    prior = torch.distributions.Normal(
        torch.zeros(len(params_to_calibrate), device=device),
        torch.ones(len(params_to_calibrate), device=device),
    )
    parameters_to_optimize = list(flow.parameters())
    true_values = [0.5, 0.5]
    print(sum([len(a) for a in parameters_to_optimize]))
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    save_dir = Path(f"/home/coml0859/bayesian_gradabm/results/nf/weight_{weight}")
    posteriors_dir = save_dir / "posteriors"
    posteriors_dir.mkdir(exist_ok=True, parents=True)

    n_samples_per_epoch = 5
    n_samples_reg = 10

    w = weight

    iterator = tqdm(range(n_epochs))
    losses = defaultdict(list)
    best_loss = np.inf

    for it in iterator:
        optimizer.zero_grad()
        forecast_loss = get_forecast_score(
            runner=runner,
            flow=flow,
            true_res=true_res,
            loss_fn=loss_fn,
            n_samples=n_samples_per_epoch,
        )
        reglrise_loss = get_regularisation(
            flow=flow, prior=prior, n_samples=n_samples_reg
        )
        loss_reg = w * reglrise_loss
        loss_reg.backward()
        loss = forecast_loss + loss_reg
        losses["forecast_train"].append(forecast_loss.item())
        losses["reglrise_train"].append(reglrise_loss.item())
        # print(loss)
        if torch.isnan(loss):
            print("loss is nan!")
            break
        # loss.backward()
        if loss.item() < best_loss:
            torch.save(flow.state_dict(), save_dir / "best_model_data.pth")
            best_loss = loss.item()
        df = pd.DataFrame(losses)
        df.to_csv(save_dir / "losses_data.csv")
        f = plot_posterior(flow, params_to_calibrate, truths=true_values, lims=(-2, 2))
        f.savefig(
            posteriors_dir / f"posterior_{it:03d}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(f)
        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fit June with flows")
    parser.add_argument("-w", "--weight")
    args = parser.parse_args()
    weight = float(args.weight)
    train(weight=weight)
