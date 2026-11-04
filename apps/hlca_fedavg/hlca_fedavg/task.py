"""Model, data loading, and the local train and evaluate loops.

The ClientApp and the ServerApp both import from here. Nothing in this module
touches Flower.

Contracts enforced at load time, because the partitions are written by hand:
feature (gene identity and order), label (one fixed class list), preprocessing
(per-client normalization only), and the split (no cell in both halves).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def gene_list_hash(genes: list[str]) -> str:
    """A short fingerprint of the gene list and its order, to compare nodes."""
    import hashlib

    return hashlib.sha1("\n".join(genes).encode()).hexdigest()[:12]


class MLP(nn.Module):
    """n_genes -> hidden -> n_classes. Logits out, paired with cross entropy."""

    def __init__(self, n_genes: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.relu(self.fc1(x))
        return self.fc2(embedding), embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_embedding(x)[0]


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


def load_partition(path: Path, cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load one partition file and return its train and validation loaders.

    The file holds raw counts over the contract's genes and an `ann_level_2`
    column. Normalizes to 1e4 counts and log1p, which is the per-client half of
    the preprocessing contract. Holds out whole donors for validation.

    Used for a client's own partition and for the central initial_training file,
    so both go through the same contract checks and the same preprocessing.
    """
    genes = load_genes(cfg["gene-list"])
    labels = list(LABELS)
    if not path.is_file():
        raise FileNotFoundError(f"No partition at {path}.")

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


def load_client(partition_id: int, cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load the partition that belongs to this Flower partition id.

    Reads `<data-dir>/client_<partition_id>.h5ad`.
    """
    data_dir = Path(os.path.expanduser(str(cfg["data-dir"])))
    return load_partition(data_dir / f"client_{partition_id}.h5ad", cfg)


def load_evaluation(
    path: str | Path, cfg: dict
) -> tuple[DataLoader, list[str], list[str]]:
    """Load the separate held-out evaluation file without splitting its cells."""
    p = Path(os.path.expanduser(str(path)))
    genes = load_genes(cfg["gene-list"])
    labels = list(LABELS)
    if not p.is_file():
        raise FileNotFoundError(f"No central evaluation partition at {p}.")

    a = ad.read_h5ad(p)
    _check_contracts(a, genes, labels, p)
    if "dataset" not in a.obs:
        raise ValueError(f"{p.name}: the evaluation partition needs an obs['dataset'] column.")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    x = a.X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)
    label_codes = {name: i for i, name in enumerate(labels)}
    y = a.obs["ann_level_2"].astype(str).map(label_codes).to_numpy(dtype=np.int64)
    dataset_names = sorted(a.obs["dataset"].astype(str).unique())
    dataset_codes = {name: i for i, name in enumerate(dataset_names)}
    dataset_ids = a.obs["dataset"].astype(str).map(dataset_codes).to_numpy(dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(np.array(x, copy=True)),
        torch.from_numpy(np.array(y, copy=True)),
        torch.from_numpy(np.array(dataset_ids, copy=True)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch-size"]), shuffle=False)
    return loader, labels, dataset_names


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


@dataclass
class EvaluationDetails:
    loss: float
    confusion: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    embeddings: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray | None
    group_confusions: dict[int, np.ndarray]


def evaluate_detailed(
    model: MLP,
    loader: DataLoader,
    n_classes: int,
    max_embedding_cells: int,
    seed: int,
) -> EvaluationDetails:
    """Evaluate every cell and retain a reproducible per-group sample for UMAP."""
    if max_embedding_cells < 0:
        raise ValueError("max_embedding_cells must be non-negative.")
    model.eval()
    n_cells = len(loader.dataset)
    data_tensors = getattr(loader.dataset, "tensors", ())
    all_groups = data_tensors[2].numpy() if len(data_tensors) > 2 else None
    rng = np.random.default_rng(seed)
    if all_groups is None:
        selected = (
            rng.choice(n_cells, min(max_embedding_cells, n_cells), replace=False)
            if max_embedding_cells
            else np.empty(0, dtype=np.int64)
        )
    else:
        selected_groups = [
            rng.choice(
                indices,
                min(max_embedding_cells, len(indices)),
                replace=False,
            )
            for group in np.unique(all_groups)
            if max_embedding_cells
            and len(indices := np.flatnonzero(all_groups == group))
        ]
        selected = (
            np.concatenate(selected_groups)
            if selected_groups
            else np.empty(0, dtype=np.int64)
        )
    selected.sort()

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    group_confusions: dict[int, np.ndarray] = {}
    embedding_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    sampled_group_parts: list[np.ndarray] = []
    total, seen = 0.0, 0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[:2]
            batch_groups = batch[2].numpy() if len(batch) > 2 else None
            logits, embeddings = model.forward_with_embedding(xb)
            total += float(F.cross_entropy(logits, yb)) * len(yb)
            seen += len(yb)
            pred = logits.argmax(1).numpy()
            true = yb.numpy()
            confusion += np.bincount(
                true * n_classes + pred, minlength=n_classes * n_classes
            ).reshape(n_classes, n_classes)
            if batch_groups is not None:
                for group in np.unique(batch_groups):
                    mask = batch_groups == group
                    group_confusions.setdefault(
                        int(group), np.zeros((n_classes, n_classes), dtype=np.int64)
                    )
                    group_confusions[int(group)] += np.bincount(
                        true[mask] * n_classes + pred[mask],
                        minlength=n_classes * n_classes,
                    ).reshape(n_classes, n_classes)

            end = offset + len(yb)
            sample_start = np.searchsorted(selected, offset)
            sample_end = np.searchsorted(selected, end)
            local_indices = selected[sample_start:sample_end] - offset
            if len(local_indices):
                embedding_parts.append(
                    embeddings[local_indices].cpu().numpy().astype(np.float32)
                )
                true_parts.append(true[local_indices])
                pred_parts.append(pred[local_indices])
                if batch_groups is not None:
                    sampled_group_parts.append(batch_groups[local_indices])
            offset = end

    diagonal = np.diag(confusion)
    tp = diagonal.copy()
    fn = confusion.sum(axis=1) - diagonal
    fp = confusion.sum(axis=0) - diagonal
    return EvaluationDetails(
        loss=total / max(seen, 1),
        confusion=confusion,
        tp=tp,
        fp=fp,
        fn=fn,
        embeddings=(
            np.concatenate(embedding_parts)
            if embedding_parts
            else np.empty((0, model.fc1.out_features), dtype=np.float32)
        ),
        y_true=np.concatenate(true_parts) if true_parts else np.empty(0, dtype=np.int64),
        y_pred=np.concatenate(pred_parts) if pred_parts else np.empty(0, dtype=np.int64),
        groups=np.concatenate(sampled_group_parts) if sampled_group_parts else None,
        group_confusions=group_confusions,
    )


def evaluate(
    model: MLP, loader: DataLoader, n_classes: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean loss and per-class true-positive, false-positive, and false-negative counts."""
    details = evaluate_detailed(
        model, loader, n_classes, max_embedding_cells=0, seed=0
    )
    return details.loss, details.tp, details.fp, details.fn


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
