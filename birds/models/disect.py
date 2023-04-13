# DiSECt physics engine

import torch
import torch.nn as nn
import tqdm
import sys
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from sgld import SGLD

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from cutting import CuttingSim
from cutting import load_settings, ConstantLinearVelocityMotion


class DiSECtRobotics(nn.Module):
    def __init__(self, settings_path, dataset='ansys', device='cuda'):
        super().__init__()
        self.settings = load_settings(settings_path)
        self.dataset = dataset
        self.gt_path = gt_path
        self.device = device
        
        self.setting.sim_duration = 0.4
        self.settings.sim_dt = 1e-5
        self.settings.initial_y = 0.075 # change this from settings file
        self.settings.velocity_y = -0.05 # change this from settings file
        
        experiment_name = f"param_inference_sgld_dt{self.settings.sim_dt}_{now.strftime('%Y%m%d-%H%M')}"
        now = datetime.now()
        self.logger = SummaryWriter(logdir=f"log/{experiment_name}")

        self.sim = CuttingSim(self.settings, dataset=self.dataset,
                 experiment_name=experiment_name, adapter=self.device, requires_grad=True)
        
        self.sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.0, self.settings.initial_y, 0.0], device=self.device),
    linear_velocity=torch.tensor([0.0, self.settings.velocity_y, 0.0], device=self.device))
        
        self.sim.cut()
                
    def forward(self):
        '''run each iteration of the simulator'''
        self.sim.motion = ConstantLinearVelocityMotion(
        initial_pos=torch.tensor(
            [0.0, settings.initial_y, 0.0], device=self.device),
        linear_velocity=torch.tensor([0.0, self.settings.velocity_y, 0.0], device=self.device))
        
        hist_knife_force = self.sim.simulate()

        return hist_knife_force
    

if __name__ == '__main__':
    settings_path = 'examples/config/ansys_sphere_apple.json'
    gt_path = 'dataset/forces/sphere_fine_resultant_force_xyz.csv'
    dataset = 'ansys'
    device = 'cuda'
    USE_SGLD = True
    
    disect = DiSECtRobotics(settings_path, dataset, device)
    sim = disect.sim
    
    opt_params = sim.init_parameters()

    sim.load_groundtruth(gt_path) # data/log_actual_2d/actual_2d_020_sgd_actual_2d.pkl

    if USE_SGLD:
        opt = SGLD(opt_params, lr=sgld_train_rate, 
                   num_burn_in_steps=burnin, use_barriers=sgld_uses_barriers,
                   noise_scaling=sgld_noise_scaling)
    else:
        opt = torch.optim.Adam(opt_params, lr=learning_rate)
        
        
    for iteration in tqdm.trange(100):
        hist_knife_force = disect()
        loss = torch.square(hist_knife_force -
                            sim.groundtruth_torch[:len(hist_knife_force)]).mean()

        print("Loss:", loss.item())

        for name, param in sim.parameters.items():
            logger.add_scalar(
                f"{name}/value", param.actual_tensor_value.mean().item(), iteration)

        disect.logger.add_scalar("loss", loss.item(), iteration)

        fig = sim.plot_simulation_results()
        fig.savefig(f"log/{experiment_name}/{experiment_name}_{iteration}.png")
        disect.logger.add_figure("simulation", fig, iteration)
        
        opt.zero_grad()
        loss.backward(retain_graph=False)

        for name, param in sim.parameters.items():
            if param.tensor.grad is None:
                print(
                    f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad N/A!')
                print(f"Iteration {iteration}: {name} has no gradient!")
            else:
                print(
                    f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad = {param.tensor.grad.mean().item()}')
                logger.add_scalar(
                    f"{name}/grad", param.tensor.grad.mean().item(), iteration)

        opt.step()

