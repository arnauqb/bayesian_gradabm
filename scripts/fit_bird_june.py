import zuko
import torch
import argparse
import datetime
import pandas as pd
import yaml

from birds import infer
from birds.models import BirdsJUNE

from grad_june import Runner


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
    flow = zuko.flows.NSF(n_parameters, 1, transforms=3, hidden_features=[128] * 3)
    flow = flow.to(device)
    return flow


def setup_prior(n_parameters, device):
    return torch.distributions.MultivariateNormal(
        loc=torch.zeros(n_parameters, device=device),
        covariance_matrix=torch.eye(n_parameters, device=device),
    )


def setup_june_config(config_path, start_date, n_days, device):
    config = yaml.safe_load(open(config_path))
    config["timer"]["initial_day"] = start_date
    config["timer"]["total_days"] = n_days
    config["system"]["device"] = device
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fit June with flows")
    parser.add_argument("--start_date", default="2020-03-01")
    parser.add_argument("--n_days", default=75)
    parser.add_argument(
        "-p", "--parameters", nargs="+", default="household company school"
    )
    parser.add_argument("--data_calibrate", nargs="+", default="cases_per_timestep")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("--data_path", default="./data/june_synth.csv")
    parser.add_argument("--results_path", default="./results_june_birds")
    parser.add_argument("--june_config", default="./configs/bird_june.yaml")
    parser.add_argument("--n_epochs", default=100)
    parser.add_argument("--n_samples_per_epoch", default=10)
    parser.add_argument("--n_samples_regularization", default=1000)
    parser.add_argument("--loss", default="LogMSELoss")
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("-w", "--weight", default=0.0)
    args = parser.parse_args()

    parameters_to_calibrate = args.parameters.split(" ")
    data_to_calibrate = args.data_calibrate.split(" ")
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
    prior = setup_prior(n_parameters, args.device)
    flow = setup_flow(n_parameters, args.device)
    obs_data = load_data(
        args.data_path,
        args.start_date,
        args.n_days,
        data_to_calibrate,
        args.device,
    )
    infer(
        model=model,
        flow=flow,
        prior=prior,
        obs_data=obs_data,
        n_epochs=args.n_samples_per_epoch,
        n_samples_per_epoch=args.n_samples_per_epoch,
        n_samples_regularization=args.n_samples_regularization,
        w=args.weight,
        save_dir=args.results_path,
        learning_rate=args.lr,
        loss=args.loss,
        save_best_posteriors=True,
        device=args.device,
        lims=None,
    )
