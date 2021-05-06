import torch
import torch.cuda
import torch.optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from ignite.engine.engine import Engine
from ignite.engine.events import Events

import numpy as np

from pathlib import Path

from vae_trainer import VAETrainer
from model import FullyConnectedVAE, CNNVAE
from evaluation import attach_metrics, create_evaluator
from handlers import EvaluationRunner, ModelSaver, StateLogger, Plotter, LogPrinter


def parse_args():
    import argparse
    global args
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
        type=Path,
        default="outputs",
        help="ouput directory")

    args = parser.parse_args()
    main()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.o.mkdir(parents=True, exist_ok=True)

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

    if args.m == "fc":
        net = FullyConnectedVAE(28 * 28, args.zdim).to(device)
    elif args.m == "cnn":
        net = CNNVAE(1, args.zdim).to(device)

    opt = torch.optim.Adam(net.parameters())
    trainer = Engine(VAETrainer(net, opt, device))
    attach_metrics(trainer)

    evaluator = create_evaluator(net, device)

    train_logger = StateLogger(trainer)
    test_logger = StateLogger(evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_logger)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        EvaluationRunner(
            evaluator,
            test_loader))
    metric_keys = ("loss", "kl_div", "recon_loss")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, test_logger)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Plotter(
        train_logger, metric_keys, args.o / "train_loss.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Plotter(
        test_logger, metric_keys, args.o / "test_loss.pdf"))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, LogPrinter(
            train_logger, metric_keys))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, LogPrinter(
            test_logger, metric_keys))
    trainer.add_event_handler(
        Events.COMPLETED, ModelSaver(
            net, args.o / "model.pt"))

    trainer.run(train_loader, max_epochs=args.e)


def mnist_transform(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(1, 28, 28) / 255


if __name__ == "__main__":
    parse_args()
