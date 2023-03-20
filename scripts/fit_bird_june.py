import zuko
import torch
import argparse
import datetime
import pandas as pd
import yaml

from birds.infer import infer, infer_fd
from birds.models import BirdsJUNE
from birds.utils import fix_seed

from grad_june import Runner


_all_parameters = [
    "seed",
    "household",
    "company",
    "school",
    "university",
    "pub",
    "grocery",
    "gym",
    "cinema",
    "visit",
    "care_visit",
    "care_home",
]

def load_data(path, start_date, n_days, data_to_calibrate, device):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    start_date_dt = pd.to_datetime(start_date).date()
    end_date_dt = start_date_dt + pd.Timedelta(days=n_days)
    df = df.loc[start_date_dt:end_date_dt]
    df = df[data_to_calibrate]
    return [
        torch.tensor(df[col], dtype=torch.float, device=device) for col in df.columns
    ]


def setup_flow(n_parameters, device):
    flow = zuko.flows.NSF(n_parameters, 1, transforms=4, hidden_features=[64] * 3)
    flow = flow.to(device)
    return flow


def setup_prior(n_parameters, device, parameter_names):
    means = []
    for name in parameter_names:
        if name == "seed":
            means.append(-3.)
        else:
            means.append(0.)
    means = torch.tensor(means, device=device)
    return torch.distributions.MultivariateNormal(
        loc=means,
        covariance_matrix=torch.eye(n_parameters, device=device),
    )


def setup_june_config(config_path, start_date, n_days, device):
    config = yaml.safe_load(open(config_path))
    config["timer"]["initial_day"] = start_date
    config["timer"]["total_days"] = n_days
    config["system"]["device"] = device
    return config


if __name__ == "__main__":
    fix_seed(0)
    parser = argparse.ArgumentParser(prog="Fit June with flows")
    parser.add_argument("--start_date", default="2020-03-01", type=str)
    parser.add_argument("--n_days", default=30, type=int)
    parser.add_argument(
        "-p", "--parameters", nargs="+", default="household school", type=str
    )
    parser.add_argument(
        "--data_calibrate", nargs="+", default="cases_per_timestep", type=str
    )
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("--data_path", default="./data/june_synth.csv", type=str)
    parser.add_argument("--results_path", default="./results_june_birds", type=str)
    parser.add_argument("--june_config", default="./configs/bird_june.yaml", type=str)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--n_samples_per_epoch", default=5, type=int)
    parser.add_argument("--n_samples_regularization", default=10000, type=int)
    parser.add_argument("--loss", default="LogMSELoss", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("-w", "--weight", default=0.01, type=float)
    parser.add_argument("--diff_mode", default="rev", type=str)
    args = parser.parse_args()

    if type(args.parameters) != list:
        parameters_to_calibrate = args.parameters.split(" ")
    else:
        parameters_to_calibrate = args.parameters
    if parameters_to_calibrate == ["all"]:
        parameters_to_calibrate = _all_parameters
    if type(args.data_calibrate) != list:
        data_to_calibrate = args.data_calibrate.split(" ")
    else:
        data_to_calibrate = args.data_calibrate
    print(f"Calibrating {parameters_to_calibrate} parameters.")
    print(f"Calibrating to {data_to_calibrate} data.")
    n_parameters = len(parameters_to_calibrate)

    config = setup_june_config(
        args.june_config, args.start_date, args.n_days, args.device
    )
    model = BirdsJUNE.from_config(
        config,
        parameters_to_calibrate=parameters_to_calibrate,
        data_to_calibrate=data_to_calibrate,
    )
    prior = setup_prior(n_parameters, args.device, parameters_to_calibrate)
    flow = setup_flow(n_parameters, args.device)
    obs_data = load_data(
        args.data_path,
        args.start_date,
        args.n_days,
        data_to_calibrate,
        args.device,
    )
    if args.diff_mode == "rev":
        infer_func = infer
    elif args.diff_mode == "fwd":
        infer_func = infer_fd
    infer_func(
        model=model,
        flow=flow,
        prior=prior,
        obs_data=obs_data,
        n_epochs=args.n_epochs,
        n_samples_per_epoch=args.n_samples_per_epoch,
        n_samples_regularization=args.n_samples_regularization,
        w=args.weight,
        save_dir=args.results_path,
        learning_rate=args.lr,
        loss=args.loss,
        plot_posteriors="never",
        device=args.device,
        true_values=None,
        lims=None,
    )
