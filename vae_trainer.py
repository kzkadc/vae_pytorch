import torch
from torch import Tensor
import torch.optim
from ignite.engine.engine import Engine

from dataclasses import dataclass
from typing import Tuple, Dict

from model import VAE
from loss import vae_loss


@dataclass
class VAETrainer:
    net: VAE
    opt: torch.optim.Optimizer
    device: torch.device

    def __call__(self,
                 engine: Engine,
                 batch: Tuple[Tensor,
                              Tensor]) -> Dict[str,
                                               Tensor]:
        self.net.train()
        self.opt.zero_grad()

        x, _ = batch
        x.to(self.device)

        z_mean, z_std = self.net.encode(x)
        z = self.net.sample(z_mean, z_std)
        x_recon = self.net.decode(z)

        loss_dict = vae_loss(x_recon, x, z_mean, z_std)
        loss_dict["loss"].backward()
        self.opt.step()

        return loss_dict
