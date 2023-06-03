from dataclasses import dataclass

import torch
from torch import Tensor
from ignite.engine.engine import Engine
from ignite.metrics import Loss

from model import VAE
from loss import recon_loss, kl_div_loss


@dataclass
class VAEEvaluator(Engine):
    net: VAE
    device: torch.device

    def __post_init__(self):
        super().__init__(self)

        x_ot = lambda d: (d["x_recon"], d["x"])
        Loss(recon_loss, x_ot).attach(self, "recon_loss")
        z_ot = lambda d: (d["z_mean"], d["z_std"])
        Loss(kl_div_loss, z_ot).attach(self, "kl_div")

    @torch.no_grad()
    def __call__(self,
                 engine: Engine,
                 batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.eval()

        x, _ = batch
        x = x.to(self.device)

        z_mean, z_std = self.net.encode(x)
        z = self.net.sample(z_mean, z_std)
        x_recon = self.net.decode(z)

        return {
            "x": x,
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_std": z_std
        }
