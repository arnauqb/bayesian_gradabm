import torch
from tqdm import tqdm
import yaml
import zuko
import corner
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from grad_june import Runner

params_to_calibrate = ["household", "school", "company"]
device = "cuda:0"
n_epochs = 500
#data_path = "/Users/arnull/code/gradabm-june/worlds/data_camden.pkl"
data_path = "/home/coml0859/data_camden.pkl"

def plot_posterior(nfm, param_names, truths=None, lims=(0, 2)):
    with torch.no_grad():
        samples  = nfm.sample((10000,))
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
    params["networks"]["household"]["log_beta"] = 0.9
    params["networks"]["school"]["log_beta"] = 0.6
    params["networks"]["company"]["log_beta"] = 0.3
    true_values = [params["networks"][name]["log_beta"] for name in params_to_calibrate]
    n_params = len(params_to_calibrate)
    params["timer"]["total_days"] = 30
    params["timer"]["initial_day"] = "2020-03-01"
    params["system"]["device"] = device
    params["data_path"] = data_path
    runner = Runner.from_parameters(params)
    return runner


def setup_flow():
    # Define flow
    n_params = len(params_to_calibrate)
    flow_unc = zuko.flows.NSF(n_params, 1, transforms=3, hidden_features=[128] * 3)
    flow_unc = flow_unc.to(device)
    return flow_unc

def run_model(runner, sample):
    sample = sample.flatten()
    for (j, name) in enumerate(params_to_calibrate):
        runner.model.infection_networks.networks[name].log_beta = sample[j]
    res, _ = runner()
    return res


def get_forecast_score(runner, flow, true_res, loss_fn, n_samples=5):
    loss = 0.0
    for i in range(n_samples):
        sample = flow.rsample()
        res = run_model(runner, sample)
        #loss_i = loss_fn(
        #        true_res["cases_per_timestep"] / runner.n_agents,
        #        res["cases_per_timestep"] / runner.n_agents,
        #)
        loss_i = 0.0
        j = 0
        for key in res:
            if "age" in key:
                loss_i += loss_fn(torch.log10(true_res[key]), torch.log10(res[key]))
                j += 1
        loss += loss_i
    return loss / n_samples


def get_regularisation(flow, prior, n_samples=5):
    samples = flow.rsample((n_samples,))
    flow_lps = flow.log_prob(samples)
    prior_lps = prior.log_prob(samples)
    kl = torch.mean(flow_lps - prior_lps)
    return kl

def train(weight):
    # Train model
    n_epochs = 10000
    runner = init_runner()
    true_res, _ = runner()
    flow_unc = setup_flow()
    prior = torch.distributions.MultivariateNormal(
        loc=torch.zeros(len(params_to_calibrate), device=device),
        covariance_matrix=torch.eye(len(params_to_calibrate), device=device),
    )
    parameters_to_optimize = list(flow_unc.parameters())
    true_values = [0.9, 0.6, 0.3]
    print(sum([len(a) for a in parameters_to_optimize]))
    optimizer = torch.optim.AdamW(parameters_to_optimize, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    save_dir = Path(f"/home/coml0859/bayesian_gradabm/results/zuko/longer/{weight}_longepochs")
    posteriors_dir = save_dir / "posteriors"
    posteriors_dir.mkdir(exist_ok=True, parents=True)

    n_samples_per_epoch = 5
    n_samples_reg = 10000

    w = torch.tensor(weight, requires_grad=True)

    iterator = tqdm(range(n_epochs))
    losses = defaultdict(list)
    best_loss = np.inf
    best_forecast_loss = np.inf

    for it in iterator:
        flow = flow_unc(torch.zeros(1, device=device))
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
        loss = forecast_loss + w * reglrise_loss
        losses["forecast_train"].append(forecast_loss.item())
        losses["reglrise_train"].append(reglrise_loss.item())
        # print(loss)
        if torch.isnan(loss):
            print("loss is nan!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_unc.parameters(), max_norm=1.0)
        if loss.item() < best_loss:
            torch.save(flow_unc.state_dict(), save_dir / "best_model_data.pth")
            best_loss = loss.item()
        if forecast_loss.item() < best_forecast_loss:
            torch.save(flow_unc.state_dict(), save_dir / "best_model_forecast.pth")
            best_forecast_loss = forecast_loss.item()

        #torch.save(flow_unc.state_dict(), save_dir / f"model_{it:03d}.pth")
        df = pd.DataFrame(losses)
        df.to_csv(save_dir / "losses_data.csv")
        try:
            f = plot_posterior(flow, params_to_calibrate, truths=true_values, lims=(-4, 4))
            f.savefig(
                posteriors_dir / f"posterior_{it:03d}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(f)
        except:
            continue
        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fit June with flows")
    parser.add_argument("-w", "--weight")
    args = parser.parse_args()
    weight = float(args.weight)
    train(weight=weight)
