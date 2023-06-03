from torch import Tensor
import torch.nn.functional as F


def recon_loss(x_recon: Tensor, x: Tensor) -> Tensor:
    return F.binary_cross_entropy(
        x_recon.flatten(start_dim=1),
        x.flatten(start_dim=1),
        reduction="none"
    ).sum(dim=1).mean()


def kl_div_loss(z_mean: Tensor, z_std: Tensor) -> Tensor:
    kl_div = -z_std.log() + (z_std.square() + z_mean.square() - 1) / 2
    kl_div = kl_div.sum(dim=1).mean()
    return kl_div
