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

from .base import InferenceEngine
from .utils import set_attribute

from .mpi_setup import mpi_rank, mpi_comm, mpi_size


class NormFlows(InferenceEngine):
    @classmethod
    def read_parameters_to_fit(cls, params):
        return params["parameters_to_fit"]

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
        return torch.optim.Adam(self.nfm.parameters(), lr=1e-4)
        config = self.training_configuration["optimizer"]
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
        return torch.nn.MSELoss(reduction="mean")
        config = self.training_configuration["loss"]
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
        true_values = [0.75, 1.25]
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

    def get_score(self, i, params, loss_fn, n_samples):
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
        loss = 0
        if mpi_rank == 0:
            params_list = [self.nfm.sample()[0][0] for i in range(n_samples)]
        else:
            params_list = None
        mpi_comm.Barrier()
        params_list = mpi_comm.bcast(params_list, root=0)
        cases = None
        for i in range(n_samples):
            if i % mpi_size == mpi_rank:
                loss_i = self.get_score(i, params_list[i], loss_fn, n_samples)
                loss += loss_i
        #mpi_comm.Barrier()
        #losses = mpi_comm.gather(loss, root=0)
        #if mpi_rank == 0:
        #    losses = [loss.to(self.device) for loss in losses]
        #    losses = torch.hstack(losses)
        #    loss = torch.sum(losses)
        #    return loss 
        #else:
        #    return None
        return loss

    def run(self):
        loss_fn = self._get_loss()
        max_iter = self.training_configuration["n_steps"]

        if mpi_rank == 0:
            self._setup_flow()
            optimizer = self._get_optimizer_and_scheduler()
            #optimizer, scheduler = self._get_optimizer_and_scheduler()

            self.posteriors_path = self.results_path / "posteriors"
            self.fits_path = self.results_path / "fits"

            if self.posteriors_path.exists():
                shutil.rmtree(self.posteriors_path)
            self.posteriors_path.mkdir(parents=True)
            if self.fits_path.exists():
                shutil.rmtree(self.fits_path)
            self.fits_path.mkdir(parents=True)

            best_loss = np.inf
            loss_hist = []

        for it in tqdm(range(max_iter)):
            if mpi_rank == 0:
                optimizer.zero_grad()

            # Sample parameters
            score = self._get_forecast_score(
                loss_fn=loss_fn,
                n_samples=self.training_configuration["n_samples"],
            )
            # self._plot_prediction(cases_mean, cases_std, self.data_observable["cases_per_timestep"], it + 1)
            # plt.close()
            # score.backward()
            if mpi_rank == 0:
                optimizer.step()
                loss_hist.append(score.item())
                if score.item() < best_loss:
                    torch.save(self.nfm.state_dict(), "./best_model.pth")
                    best_loss = score.item()
                self._plot_posterior(it + 1)
                plt.close()
                self._plot_loss_hist(loss_hist)
                plt.close()

            # Log loss
        loss_hist = np.array(loss_hist)
        return
