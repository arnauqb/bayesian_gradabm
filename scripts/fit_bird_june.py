import zuko
import torch
import argparse
import datetime
import pandas as pd
import yaml

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

from birds.infer import infer
from birds.models import BirdsJUNE
from birds.utils import fix_seed

from grad_june import Runner

_all_no_seed_parameters = [
    "beta_household",
    "beta_company",
    "beta_school",
    "beta_university",
    "beta_pub",
    "beta_grocery",
    "beta_gym",
    "beta_cinema",
    "beta_visit",
    "beta_care_visit",
    "beta_care_home",
    "sd_company",
    "sd_school",
    "sd_care_home",
    "sd_care_visit",
    "sd_grocery",
]

_all_parameters = ["seed"] + _all_no_seed_parameters

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
    #flow = zuko.flows.MAF(n_parameters, 1, transforms=10, hidden_features=[50] * 2)
    flow = flow.to(device)
    return flow


def setup_prior(n_parameters, device, parameter_names):
    means = []
    stds = []
    for name in parameter_names:
        if name == "seed":
            means.append(-3.0)
            stds.append(1.0)
        else:
            means.append(0.0)
            stds.append(1.0)
    means = torch.tensor(means, device=device)
    stds = torch.tensor(stds, device=device)
    return torch.distributions.MultivariateNormal(
        loc=means,
        covariance_matrix= stds * torch.eye(n_parameters, device=device),
    )


def setup_june_config(config_path, start_date, n_days, device):
    config = yaml.safe_load(open(config_path))
    config["timer"]["initial_day"] = start_date
    config["timer"]["total_days"] = n_days
    config["system"]["device"] = device
    return config


if __name__ == "__main__":
    #fix_seed(0)
    parser = argparse.ArgumentParser(prog="Fit June with flows")
    parser.add_argument("--start_date", default="2020-03-01", type=str)
    parser.add_argument("--n_days", default=30, type=int)
    parser.add_argument(
        "-p", "--parameters", nargs="+", default="household school", type=str
    )
    parser.add_argument(
        "--data_calibrate", nargs="+", default="cases_per_timestep", type=str
    )
    parser.add_argument("-d", "--device", default="cpu", type=str, nargs = "+")
    parser.add_argument("--data_path", default="./data/june_synth.csv", type=str)
    parser.add_argument("--results_path", default="./results_june_birds", type=str)
    parser.add_argument("--june_config", default="./configs/bird_june.yaml", type=str)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--n_samples_per_epoch", default=5, type=int)
    parser.add_argument("--n_samples_regularization", default=10000, type=int)
    parser.add_argument("--loss", default="LogMSELoss", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("-w", "--weight", default=0.01, type=float)
    parser.add_argument("--diff_mode", default="reverse", type=str)
    parser.add_argument("--chunk_size", default=0, type=int)
    parser.add_argument("--load_model", default=None, type=str)
    args = parser.parse_args()

    if type(args.parameters) != list:
        parameters_to_calibrate = args.parameters.split(" ")
    else:
        parameters_to_calibrate = args.parameters
    if parameters_to_calibrate == ["all"]:
        parameters_to_calibrate = _all_parameters
    elif parameters_to_calibrate == ["all_no_seed"]:
        parameters_to_calibrate = _all_no_seed_parameters
    if type(args.data_calibrate) != list:
        data_to_calibrate = args.data_calibrate.split(" ")
    else:
        data_to_calibrate = args.data_calibrate
    if args.chunk_size == 0:
        jacobian_chunk_size = None
    else:
        jacobian_chunk_size = args.chunk_size
    device = args.device[mpi_rank]
    print(f"Rank {mpi_rank} using device {device}")
    if mpi_rank == 0:
        print(f"Calibrating {parameters_to_calibrate} parameters.")
        print(f"Calibrating to {data_to_calibrate} data.")
        print(f"Regularization parameter set to {args.weight}")
        print(f"Saving results to {args.results_path}")
        print(f"Pre-loading model dict {args.load_model}")
        print(f"Initial date is {args.start_date}")
    n_parameters = len(parameters_to_calibrate)

    config = setup_june_config(
        args.june_config, args.start_date, args.n_days, device
    )
    model = BirdsJUNE.from_config(
        config,
        parameters_to_calibrate=parameters_to_calibrate,
        data_to_calibrate=data_to_calibrate,
    )
    prior = setup_prior(n_parameters, device, parameters_to_calibrate)
    flow = setup_flow(n_parameters, device)
    obs_data = load_data(
        args.data_path,
        args.start_date,
        args.n_days,
        data_to_calibrate,
        device,
    )
    infer(
        model=model,
        flow=flow,
        prior=prior,
        obs_data=obs_data,
        diff_mode=args.diff_mode,
        jacobian_chunk_size=jacobian_chunk_size,
        n_epochs=args.n_epochs,
        n_samples_per_epoch=args.n_samples_per_epoch,
        n_samples_regularization=args.n_samples_regularization,
        w=args.weight,
        save_dir=args.results_path,
        learning_rate=args.lr,
        loss=args.loss,
        preload_model_path = args.load_model,
        plot_posteriors="never",
        device=device,
        true_values=None,
        lims=None,
    )
