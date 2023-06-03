from pathlib import Path

import torch
import torch.cuda
import torch.optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from ignite.engine.events import Events

import numpy as np

from vae_trainer import VAETrainer
from model import FullyConnectedVAE, CNNVAE
from evaluation import VAEEvaluator
from handlers import EvaluationRunner, ModelSaver, StateLogger, Plotter, LogPrinter


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=20, help="epoch")
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument(
        "--zdim",
        type=int,
        default=20,
        help="number of dimensions of latent space")
    parser.add_argument(
        "-m",
        required=True,
        choices=[
            "fc",
            "cnn"],
        help="model architecture")
    parser.add_argument(
        "-o",
        default="outputs",
        help="ouput directory")

    args = parser.parse_args()
    main(args)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU mode")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    Path(args.o).mkdir(parents=True, exist_ok=True)

    mnist_transform = lambda x: np.asarray(
        x, dtype=np.float32).reshape(1, 28, 28) / 255
    train_dataset = MNIST(
        root=".",
        download=True,
        train=True,
        transform=mnist_transform)

    test_dataset = MNIST(
        root=".",
        download=True,
        train=False,
        transform=mnist_transform)

    train_loader = DataLoader(train_dataset, args.b, shuffle=True)
    test_loader = DataLoader(test_dataset, args.b)

    match args.m:
        case "fc":
            net = FullyConnectedVAE(28 * 28, args.zdim).to(device)
        case "cnn":
            net = CNNVAE(1, args.zdim).to(device)
        case _:
            raise ValueError(f"Invalid model: {args.m!r}")

    opt = torch.optim.Adam(net.parameters())
    trainer = VAETrainer(net, opt, device)

    evaluator = VAEEvaluator(net, device)

    train_logger = StateLogger(trainer)
    test_logger = StateLogger(evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_logger)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        EvaluationRunner(evaluator, test_loader))
    metric_keys = ("kl_div", "recon_loss")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, test_logger)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Plotter(
        train_logger, metric_keys, Path(args.o, "train_loss.pdf")))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Plotter(
        test_logger, metric_keys, Path(args.o, "test_loss.pdf")))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, LogPrinter(train_logger, metric_keys))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, LogPrinter(test_logger, metric_keys))
    trainer.add_event_handler(
        Events.COMPLETED, ModelSaver(net, Path(args.o, "model.pt")))

    trainer.run(train_loader, max_epochs=args.e)


if __name__ == "__main__":
    parse_args()
