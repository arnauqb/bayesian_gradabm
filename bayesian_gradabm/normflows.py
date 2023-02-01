import normflows as nf
import corner
import numpy as np
import torch
import shutil
import pandas as pd
from collections import defaultdict
from time import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from .base import InferenceEngine


class NormFlows(InferenceEngine):
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
        self.flow = nf.NormalizingFlow(q0=q0, flows=flows)
        self.flow = self.flow.to(self.device)

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
        optimizer = optimizer_class(self.flow.parameters(), **config)
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
        posteriors = self.flow.sample(10000)
        samples = posteriors[0].cpu().detach().numpy()
        labels = [name.split(".")[-2] for name in self.priors]
        true_values = [1.0, 1.0]
        f = corner.corner(
            samples,
            labels=labels,
            smooth=2,
            truths=true_values,
            show_titles=True,
            bins=30,
            range=[(0, 2) for i in range(len(true_values))],
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

    def get_forecast_score(self, flow, true_data, loss_fn, n_samples=5):
        loss = 0.0
        for i in range(n_samples):
            sample, lp = flow.sample()
            samples_dict = {}
            for i, key in enumerate(self.priors):
                samples_dict[key] = sample[0][i]
            cases_per_timestep = (
                self.evaluate(samples_dict)["cases_per_timestep"] / self.runner.n_agents
            )
            loss += loss_fn(cases_per_timestep, true_data)
        return loss / n_samples

    def get_regularisation(self, flow, n_samples=5):
        samples, flow_lps = flow.sample(n_samples)
        prior_lps = torch.zeros(samples.shape)
        for i, key in enumerate(self.priors):
            prior_dist = self.priors[key]
            prior_lps[:, i] = prior_dist.log_prob(samples[:, i])
        prior_lps = prior_lps.sum(1)
        kl = torch.mean(flow_lps - prior_lps)
        return kl

    def run(self):
        # Train model
        self._setup_flow()
        n_params_train = sum([len(a) for a in list(self.flow.parameters())])
        print(f"Training {n_params_train} parameters")
        optimizer, scheduler = self._get_optimizer_and_scheduler()
        loss_fn = self._get_loss()
        n_epochs = self.training_configuration["n_epochs"]
        n_batch = self.training_configuration["n_batch"]
        n_samples_reg = 10
        w = 0.0
        true_data = self.observed_data["cases_per_timestep"] / self.runner.n_agents
        iterator = tqdm(range(n_epochs))
        losses = defaultdict(list)
        best_loss = torch.inf
        for it in iterator:
            optimizer.zero_grad()
            forecast_loss = self.get_forecast_score(
                flow=self.flow,
                true_data=true_data,
                loss_fn=loss_fn,
                n_samples=n_batch,
            )
            reglrise_loss = self.get_regularisation(
                flow=self.flow, n_samples=n_samples_reg
            )
            loss = forecast_loss + w * reglrise_loss
            losses["forecast_train"].append(forecast_loss.item())
            losses["reglrise_train"].append(reglrise_loss.item())
            if torch.isnan(loss):
                print("loss is nan!")
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_forecast_loss = self.get_forecast_score(
                    flow=self.flow,
                    true_data=true_data,
                    loss_fn=loss_fn,
                    n_samples=n_batch,
                )
                val_reglrise_loss = self.get_regularisation(
                    flow=self.flow, n_samples=n_samples_reg
                )
                val_loss = val_forecast_loss + w * val_reglrise_loss

                losses["forecast_val"].append(val_forecast_loss.item())
                losses["reglrise_val"].append(val_reglrise_loss.item())

                if val_loss.item() < best_loss:
                    torch.save(
                        self.flow.state_dict(), self.results_path / "best_model.pth"
                    )
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
            df.to_csv(self.results_path / "losses.csv")
