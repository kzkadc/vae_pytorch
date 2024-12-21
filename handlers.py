from typing import Any
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import copy

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from ignite.engine.engine import Engine

import matplotlib.pyplot as plt


@dataclass
class EvaluationRunner:
    evaluator: Engine
    dataloader: DataLoader

    def __call__(self, engine: Engine):
        if self.evaluator is not None:
            self.evaluator.run(self.dataloader)


@dataclass
class StateLogger:
    engine: Engine

    def __post_init__(self):
        self.metrics: list[dict[str, Any]] = []
        self.epochs: list[int] = []

    def __call__(self, engine: Engine):
        self.epochs.append(engine.state.epoch)
        self.metrics.append(copy.copy(self.engine.state.metrics))


@dataclass
class Plotter:
    logger: StateLogger
    keys: Iterable[str]
    path: Path

    def __call__(self, engine: Engine):
        plt.figure()

        for k in self.keys:
            y = [s[k] for s in self.logger.metrics]
            plt.plot(self.logger.epochs, y, label=k)

        plt.legend()
        plt.xlabel("epoch")
        plt.savefig(str(self.path))
        plt.close()


@dataclass
class LogPrinter:
    logger: StateLogger
    keys: Iterable[str]

    def __call__(self, engine: Engine):
        s = []
        for k in self.keys:
            s.append(f"{k}={self.logger.metrics[-1][k]}")

        print(f"epoch {engine.state.epoch}: {', '.join(s)}")


@dataclass
class ModelSaver:
    net: nn.Module
    model_path: Path

    def __call__(self, engine: Engine):
        torch.save(self.net.state_dict(), str(self.model_path))
