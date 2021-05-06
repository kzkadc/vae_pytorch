import torch
from torch import Tensor
from ignite.engine.engine import Engine
from ignite.metrics import Average

from dataclasses import dataclass
from typing import Tuple, Dict

from model import VAE
from loss import vae_loss


@dataclass
class VAEEvaluator:
    net: VAE
    device: torch.device

    def __call__(self,
                 engine: Engine,
                 batch: Tuple[Tensor,
                              Tensor]) -> Dict[str,
                                               Tensor]:
        self.net.eval()

        x, _ = batch
        x = x.to(self.device)

        with torch.no_grad():
            z_mean, z_std = self.net.encode(x)
            z = self.net.sample(z_mean, z_std)
            x_recon = self.net.decode(z)

            loss_dict = vae_loss(x_recon, x, z_mean, z_std)

        return loss_dict


def attach_metrics(engine: Engine):
    Average(output_transform=lambda d: d["loss"]).attach(engine, "loss")
    Average(output_transform=lambda d: d["kl_div"]).attach(engine, "kl_div")
    Average(
        output_transform=lambda d: d["recon_loss"]).attach(
        engine,
        "recon_loss")


def create_evaluator(net: VAE, device: torch.device) -> Engine:
    evaluator = Engine(VAEEvaluator(net, device))
    attach_metrics(evaluator)

    return evaluator
