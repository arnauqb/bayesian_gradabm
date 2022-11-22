import normflows as nf
import corner
import numpy as np
import torch
import shutil
from time import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp

from .base import InferenceEngine
from .utils import set_attribute

from .mpi_setup import mpi_rank, mpi_comm, mpi_size


class NormFlows(InferenceEngine):
    @classmethod
    def read_parameters_to_fit(cls, params):
        return params["parameters_to_fit"]

    def _make_dirs(self):
        self.posteriors_path = self.results_path / "posteriors"
        self.fits_path = self.results_path / "fits"

        if self.posteriors_path.exists():
            shutil.rmtree(self.posteriors_path)
        self.posteriors_path.mkdir(parents=True)
        if self.fits_path.exists():
            shutil.rmtree(self.fits_path)
        self.fits_path.mkdir(parents=True)

    def _setup_flow(self):
        flow_config = deepcopy(self.training_configuration["flow"])
        K = flow_config.pop("K")
        flows = []
        for i in range(K):
            for flow_name in flow_config:
                flow_class = getattr(nf.flows, flow_name)
                flow = flow_class(**flow_config[flow_name])
                flows.append(flow)
        q0 = nf.distributions.DiagGaussian(len(self.priors), trainable=False)
        self.nfm = nf.NormalizingFlow(q0=q0, flows=flows)
        self.nfm = self.nfm.to(self.device)

    def _get_optimizer_and_scheduler(self):
        config = deepcopy(self.training_configuration["optimizer"])
        optimizer_type = config.pop("type")
        if "milestones" in config:
            milestones = config.pop("milestones")
        else:
            milestones = []
        if "gamma" in config:
            gamma = config.pop("gamma")
        else:
            gamma = 1.0
        optimizer_class = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_class(self.nfm.parameters(), **config)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        return optimizer, scheduler

    def _get_loss(self):
        config = deepcopy(self.training_configuration["loss"])
        loss_class = getattr(torch.nn, config.pop("type"))
        loss = loss_class(**config)
        return loss

    def _plot_posterior(self, it):
        posteriors = self.nfm.sample(10000)
        samples = posteriors[0].cpu().detach().numpy()
        samples = np.where(~np.isnan(samples), samples, 0)
        samples = np.where(~np.isinf(samples), samples, 0)
        samples = np.where(np.abs(samples) < 5, samples, 0)
        labels = [name.split(".")[-2] for name in self.priors]
        true_values = [1.5, 2.0]
        f = corner.corner(
            samples,
            labels=labels,
            smooth=2,
            truths=true_values,
            show_titles=True,
            bins=30,
            range=[(-4, 4) for i in range(len(true_values))],
        )
        f.savefig(
            self.results_path / f"posteriors/posterior_{it:03d}.png",
            dpi=150,
            facecolor="white",
        )
        return

    def _plot_prediction(self, cases_mean, cases_std, data, it):
        dates = data["dates"]
        f, ax = plt.subplots()
        cases_mean = cases_mean.cpu()
        cases_std = cases_std.cpu()
        ax.plot(dates, data["cases_per_timestep"].cpu(), color="black", label="data")
        ax.plot(dates, cases_mean, color="C0", label="prediction")
        ax.fill_between(
            dates,
            cases_mean - cases_std,
            cases_mean + cases_std,
            color="C0",
            alpha=0.5,
            linewidth=0,
        )
        ax.set_yscale("log")
        ax.legend()
        ax.set_ylabel("Cases")
        f.autofmt_xdate()
        f.savefig(
            self.results_path / f"fits/fit_{it:03d}.png", dpi=150, facecolor="white"
        )
        return

    def _plot_loss_hist(self, loss_hist):
        f, ax = plt.subplots()
        ax.plot(loss_hist)
        ax.set_ylabel("loss")
        ax.set_xlabel("iteration")
        ax.set_yscale("log")
        f.savefig(self.results_path / "loss.png", dpi=150)
        return

    def _get_score(self, params, loss_fn, n_samples):
        params = torch.clip(params, min=-5, max=5)
        for i, name in enumerate(self.priors):
            set_attribute(self.runner, name, params[i].to(self.device))
        results, _ = self.runner()
        for key in self.data_observable:
            time_stamps = self.data_observable[key]["time_stamps"]
            if time_stamps == "all":
                time_stamps = range(len(results["dates"]))
            y = results[key][time_stamps]
            y_obs = self.observed_data[key][time_stamps]
            loss_i = loss_fn(y, y_obs) / n_samples
        loss_i.backward()
        return loss_i

    def _get_forecast_score(self, loss_fn, n_samples=5):
        # mpi_comm.Barrier()
        # params_list = mpi_comm.bcast(params_list, root=0)
        # params_list, _ = self.nfm.sample(n_samples)
        for i in range(n_samples):
            if i % mpi_size == mpi_rank:
                params = params_list[i]
                loss_i = self._get_score(params, loss_fn, n_samples)
                loss += loss_i
        # mpi_comm.Barrier()
        # losses = mpi_comm.gather(loss, root=0)
        # if mpi_rank == 0:
        #    losses = [loss.to(self.device) for loss in losses]
        #    losses = torch.hstack(losses)
        #    loss = torch.sum(losses)
        #    return loss
        # else:
        #    return None
        return loss

    def compute_score(self, rank, size, loss_fn, n_samples):
        loss = 0
        params_to_run = []
        for i in range(n_samples):
            dest_rank = i % size
            if rank == 0:
                params, _ = self.nfm.sample()
                params = params[0].to("cpu")
                params = torch.tensor([1.,1.])
                print(params)
                dist.send(tensor=params, dst=dest_rank)
            elif dest_rank == rank:
                params = torch.zeros(2, )
                print(params)
                dist.recv(tensor=params, src=0)
                print(f"I am rank {rank} and have params {params}")


    def _setup_rank(self, rank, size):
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=size)

    def run_rank(self, rank, size):
        self._setup_rank(rank, size)
        max_iter = self.training_configuration["n_steps"]
        loss_fn = self._get_loss()
        n_samples = 10
        if rank == 0:
            self._make_dirs()
            self._setup_flow()
            optimizer, scheduler = self._get_optimizer_and_scheduler()
            best_loss = np.inf
            loss_hist = []

        for it in tqdm(range(max_iter)):
            if rank == 0:
                optimizer.zero_grad()

            score = self.compute_score(rank, size, loss_fn, n_samples)
            if rank == 0:
                optimizer.step()
                loss_hist.append(score.item())
                if score.item() < best_loss:
                    torch.save(self.nfm.state_dict(), "./best_model.pth")
                    best_loss = score.item()

    def run(self):
        size = 2
        processes = []
        for rank in range(size):
            p = mp.Process(target=self.run_rank, args=(rank, size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
