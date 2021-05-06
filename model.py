import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from typing import Callable, Final, Tuple


class VAE(nn.Module):
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(
            self,
            z_mean: Tensor,
            z_std: Tensor) -> Tensor:
        r = torch.normal(torch.zeros_like(z_mean))
        z = z_mean + r * z_std
        return z

    def forward(self, x: Tensor) -> Tensor:
        z_mean, z_std = self.encode(x)
        z = self.sample(z_mean, z_std)
        recon = self.decode(z)
        return recon


class FullyConnectedVAE(VAE):
    def __init__(self, xdim: int, zdim: int):
        super().__init__()
        HID_DIM: Final[int] = 100
        self.enc_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(xdim, HID_DIM, bias=False),
            #    nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM, bias=False),
            #    nn.BatchNorm1d(HID_DIM),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(HID_DIM, zdim)
        self.enc_std = nn.Linear(HID_DIM, zdim)

        self.dec = nn.Sequential(
            nn.Linear(zdim, HID_DIM, bias=False),
            #    nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM, bias=False),
            #    nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, xdim),
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc_fc(x)
        z_mean = self.enc_mean(h)
        z_std = F.softplus(self.enc_std(h))
        return z_mean, z_std

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)


class CNNVAE(VAE):
    def __init__(self, in_channels: int, zdim: int):
        super().__init__()

        N: Final[int] = 32
        kwargs1 = {"kernel_size": 4, "stride": 2, "padding": 1}
        kwargs2 = {"kernel_size": 3, "stride": 1, "padding": 0}
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, N, bias=False, **kwargs1),  # 14
            # nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.Conv2d(N, N * 2, bias=False, **kwargs1),  # 7
            #nn.BatchNorm2d(N * 2),
            nn.ReLU(),
            nn.Conv2d(N * 2, N * 4, bias=False, **kwargs2),    # 5
            #nn.BatchNorm2d(N * 4),
            nn.ReLU()
        )
        self.enc_mean = nn.Sequential(
            nn.Conv2d(N * 4, N * 8, bias=False, **kwargs2),    # 3
            #nn.BatchNorm2d(N * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(N * 8, zdim)
        )
        self.enc_std = nn.Sequential(
            nn.Conv2d(N * 4, N * 8, bias=False, **kwargs2),     # 3
            #nn.BatchNorm2d(N * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(N * 8, zdim)
        )

        self.dec = nn.Sequential(
            nn.Linear(zdim, N * 8, bias=False),
            Lambda(lambda x: x.reshape(-1, N * 8, 1, 1)),   # 1
            nn.ZeroPad2d(1),  # 3
            #nn.BatchNorm2d(N * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(N * 8, N * 4, **kwargs2),    # 5
            #nn.BatchNorm2d(N * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(N * 4, N * 2, bias=False, **kwargs2),      # 7
            #nn.BatchNorm2d(N * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(N * 2, N, bias=False, **kwargs1),   # 14
            # nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.ConvTranspose2d(N, in_channels, **kwargs1),    # 28
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc_conv(x)
        z_mean = self.enc_mean(h)
        z_std = F.softplus(self.enc_std(h))
        return z_mean, z_std

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)


class Lambda(nn.Module):
    def __init__(self, func: Callable[[Tensor], Tensor]):
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)
