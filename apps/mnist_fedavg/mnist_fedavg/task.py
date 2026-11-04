"""Model, data partitioning, and the local train/test loops.

Shared by the ClientApp (every SuperNode) and the ServerApp (central evaluation).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class MLP(nn.Module):
    """784 -> 128 -> 10, log-probabilities out (paired with nll_loss, as in the original)."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


@lru_cache(maxsize=2)
def _load_mnist(data_root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Download MNIST once per machine and return normalized tensors (N, 784), (N,)."""
    from torchvision.datasets import MNIST

    root = Path(os.path.expanduser(data_root))
    root.mkdir(parents=True, exist_ok=True)
    ds = MNIST(root=str(root), train=train, download=True)
    x = (ds.data.float() / 255.0 - MNIST_MEAN) / MNIST_STD
    return x.reshape(len(x), -1), ds.targets.clone()


def partition_indices(
    labels: np.ndarray,
    partition_id: int,
    num_partitions: int,
    non_iid: bool,
    shards_per_client: int,
    seed: int,
) -> np.ndarray:
    """Indices of this client's share of the training set.

    Deterministic given the seed, so every node computes the same global split
    and takes a disjoint slice.
    IID: random permutation split into equal parts.
    Non-IID (McMahan et al. 2017): sort by label, cut into
    num_partitions * shards_per_client shards, deal shards_per_client to each client.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if not non_iid:
        return np.array_split(rng.permutation(n), num_partitions)[partition_id]

    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, num_partitions * shards_per_client)
    shard_ids = rng.permutation(len(shards))
    mine = shard_ids[partition_id * shards_per_client : (partition_id + 1) * shards_per_client]
    return np.concatenate([shards[s] for s in mine])


def load_client_data(
    partition_id: int,
    num_partitions: int,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """Train and local-validation loaders for one client, plus the digits it holds."""
    x, y = _load_mnist(cfg["data-root"], True)
    idx = partition_indices(
        y.numpy(),
        partition_id,
        num_partitions,
        bool(cfg["non-iid"]),
        int(cfg["shards-per-client"]),
        int(cfg["seed"]),
    )
    rng = np.random.default_rng(int(cfg["seed"]) + 1000 + partition_id)
    idx = rng.permutation(idx)
    n_val = int(len(idx) * float(cfg["val-fraction"]))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    bs = int(cfg["batch-size"])
    g = torch.Generator().manual_seed(int(cfg["seed"]) + partition_id)
    train = DataLoader(TensorDataset(x[train_idx], y[train_idx]), batch_size=bs, shuffle=True, generator=g)
    val = DataLoader(TensorDataset(x[val_idx], y[val_idx]), batch_size=512)
    digits = sorted(int(d) for d in torch.unique(y[idx]))
    return train, val, digits


def load_test_data(data_root: str) -> DataLoader:
    """Full 10k MNIST test set, for central evaluation on the server."""
    x, y = _load_mnist(data_root, False)
    return DataLoader(TensorDataset(x, y), batch_size=1024)


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, momentum: float) -> list[float]:
    """Local SGD. Returns the mean training loss of each epoch."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    epoch_losses = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.nll_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            count += len(yb)
        epoch_losses.append(total / count)
    return epoch_losses


@torch.no_grad()
def test(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Mean loss and accuracy."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        out = model(xb)
        total_loss += F.nll_loss(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
        count += len(yb)
    return total_loss / count, correct / count
