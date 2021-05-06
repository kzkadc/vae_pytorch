import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Dict


def vae_loss(x_recon: Tensor, x: Tensor, z_mean: Tensor,
             z_std: Tensor) -> Dict[str, Tensor]:
    recon_loss = F.binary_cross_entropy(
        x_recon.flatten(start_dim=1),
        x.flatten(start_dim=1), reduction="none").sum(dim=1).mean()
    kl_div = -z_std.log() + (z_std.square() + z_mean.square() - 1) / 2
    kl_div = kl_div.sum(dim=1).mean()

    return {
        "loss": recon_loss + kl_div,
        "kl_div": kl_div,
        "recon_loss": recon_loss
    }
