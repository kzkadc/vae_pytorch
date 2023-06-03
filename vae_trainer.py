from dataclasses import dataclass

import torch
from torch import Tensor
import torch.optim
from ignite.engine.engine import Engine
from ignite.metrics import Loss

from model import VAE
from loss import recon_loss, kl_div_loss


@dataclass
class VAETrainer(Engine):
    net: VAE
    opt: torch.optim.Optimizer
    device: torch.device

    def __post_init__(self):
        super().__init__(self)

        x_ot = lambda d: (d["x_recon"], d["x"])
        Loss(recon_loss, x_ot).attach(self, "recon_loss")
        z_ot = lambda d: (d["z_mean"], d["z_std"])
        Loss(kl_div_loss, z_ot).attach(self, "kl_div")

    def __call__(self,
                 engine: Engine,
                 batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.train()
        self.opt.zero_grad()

        x, _ = batch
        x = x.to(self.device)

        z_mean, z_std = self.net.encode(x)
        z = self.net.sample(z_mean, z_std)
        x_recon = self.net.decode(z)

        loss = recon_loss(x_recon, x) + kl_div_loss(z_mean, z_std)
        loss.backward()
        self.opt.step()  # pylint: disable=E1120

        return {
            "x": x,
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_std": z_std
        }
