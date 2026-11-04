"""Model, data loading, and the local train and evaluate loops.

The ClientApp and the ServerApp both import from here. Nothing in this module
touches Flower.

Contracts enforced at load time, because the partitions are written by hand:
feature (gene identity and order), label (one fixed class list), preprocessing
(per-client normalization only), and the split (no cell in both halves).
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The label contract. The order fixes the output width, and a client missing a
# class still returns a weight matrix of this width. Mesothelium and
# Hematopoietic stem cells are left out on purpose: 66 and 44 cells across the
# whole federation, too few to learn or to score, and as classes they would own
# two elevenths of macro F1.
LABELS: tuple[str, ...] = (
    "Airway epithelium",
    "Alveolar epithelium",
    "Blood vessels",
    "Fibroblast lineage",
    "Lymphatic EC",
    "Lymphoid",
    "Myeloid",
    "Smooth muscle",
    "Submucosal Gland",
)


def load_genes(path: str | Path) -> list[str]:
    """The fixed gene list, in the order every client must present."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_file():
        raise FileNotFoundError(
            f"No gene list at {p}. It is the feature contract: one gene symbol "
            "per line, in the order the partitions use."
        )
    genes = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not genes:
        raise ValueError(f"The gene list at {p} is empty.")
    return genes


class MLP(nn.Module):
    """n_genes -> hidden -> n_classes. Logits out, paired with cross entropy."""

    def __init__(self, n_genes: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def build_model(cfg: dict) -> MLP:
    """One place that fixes the parameter shapes, so every node agrees."""
    genes = load_genes(cfg["gene-list"])
    return MLP(len(genes), int(cfg["hidden-size"]), len(LABELS))


def _check_contracts(a: ad.AnnData, genes: list[str], labels: list[str], path: Path) -> None:
    """Raise on the first contract violation, naming the file and the detail.

    A reordered gene list is the dangerous case: the matrix widths still agree,
    so training and averaging run and the result is meaningless.
    """
    have = list(a.var_names)
    if len(have) != len(genes):
        raise ValueError(f"{path.name}: {len(have)} genes, the contract has {len(genes)}.")
    if have != genes:
        first = next(i for i, (x, y) in enumerate(zip(have, genes)) if x != y)
        raise ValueError(
            f"{path.name}: gene order differs from the contract at index {first}: "
            f"file has {have[first]}, contract has {genes[first]}. "
            "Reindex the partition, do not reorder it here."
        )
    unknown = sorted(set(a.obs["ann_level_2"].astype(str)) - set(labels))
    if unknown:
        raise ValueError(f"{path.name}: labels outside the contract: {unknown}.")
    x = a.X[: min(1000, a.n_obs)]
    values = x.data if hasattr(x, "data") else np.asarray(x).ravel()
    if values.size and (values.min() < 0 or not np.allclose(values, np.rint(values))):
        raise ValueError(
            f"{path.name}: X is not raw counts. The partitions carry counts, and "
            "normalization runs per client."
        )


def _split_indices(a: ad.AnnData, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole donors where the column exists, else whole cells.

    A donor split is the honest one: cells from one donor are not independent,
    so a cell-level split reports a number the model cannot repeat on a new
    patient.
    """
    rng = np.random.default_rng(seed)
    if "donor_id" in a.obs:
        donors = a.obs["donor_id"].astype(str).to_numpy()
        order = rng.permutation(np.unique(donors))
        target = val_fraction * len(donors)
        held: list[str] = []
        held_cells = 0
        for d in order:
            if held_cells >= target or len(held) == len(order) - 1:
                break
            held.append(d)
            held_cells += int((donors == d).sum())
        is_val = np.isin(donors, held)
    else:
        is_val = np.zeros(a.n_obs, dtype=bool)
        is_val[rng.permutation(a.n_obs)[: int(round(val_fraction * a.n_obs))]] = True
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("The train and the validation split overlap.")
    return train_idx, val_idx


def load_client(partition_id: int, cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load one client's partition and return its train and validation loaders.

    Reads `<data-dir>/client_<partition_id>.h5ad`, which holds raw counts over
    the contract's genes and an `ann_level_2` column. Normalizes to 1e4 counts
    and log1p per client, which is the per-client half of the preprocessing
    contract.
    """
    genes = load_genes(cfg["gene-list"])
    labels = list(LABELS)
    path = Path(os.path.expanduser(str(cfg["data-dir"]))) / f"client_{partition_id}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(f"No partition for client {partition_id} at {path}.")

    a = ad.read_h5ad(path)
    _check_contracts(a, genes, labels, path)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    x = a.X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)
    code = {name: i for i, name in enumerate(labels)}
    y = a.obs["ann_level_2"].astype(str).map(code).to_numpy(dtype=np.int64)

    train_idx, val_idx = _split_indices(a, float(cfg["val-fraction"]), int(cfg["seed"]))
    batch = int(cfg["batch-size"])
    # np.array(copy=True): a view of the AnnData buffer is read only, and torch
    # refuses to wrap that without a warning.
    features = torch.from_numpy(np.array(x, copy=True))
    targets = torch.from_numpy(np.array(y, copy=True))

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        sub = TensorDataset(features[idx], targets[idx])
        return DataLoader(sub, batch_size=batch, shuffle=shuffle)

    return loader(train_idx, True), loader(val_idx, False), labels


def train(model: MLP, loader: DataLoader, epochs: int, lr: float, momentum: float) -> list[float]:
    """Train in place. Returns the mean loss of each epoch."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    losses = []
    for _ in range(epochs):
        total, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            seen += len(yb)
        losses.append(total / max(seen, 1))
    return losses


def evaluate(model: MLP, loader: DataLoader, n_classes: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean loss and the per-class true positive, false positive and
    false negative counts.

    Counts, not a score: macro F1 is not linear in the clients, so the server
    sums these and computes one macro F1 from the totals.
    """
    model.eval()
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    total, seen = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total += float(F.cross_entropy(logits, yb)) * len(yb)
            seen += len(yb)
            pred = logits.argmax(1).numpy()
            true = yb.numpy()
            for c in range(n_classes):
                tp[c] += int(((pred == c) & (true == c)).sum())
                fp[c] += int(((pred == c) & (true != c)).sum())
                fn[c] += int(((pred != c) & (true == c)).sum())
    return total / max(seen, 1), tp, fp, fn


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    """Unweighted mean F1 over the classes that appear in the truth.

    A class with no true cell anywhere is undefined, not zero, so it is left
    out rather than scored as a failure.
    """
    present = (tp + fn) > 0
    if not present.any():
        return 0.0
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0)
    return float(f1[present].mean())
