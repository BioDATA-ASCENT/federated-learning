"""Model, data loading, and the local train and test loops for the logistic
regression app.

The ClientApp, the ServerApp and baseline.py all import from here. Nothing in
this module touches Flower.

Contracts enforced at load time, because the partitions are written by hand:
feature (gene identity and order), label (one fixed class list), preprocessing
(per-client normalization only, no per-client scaling), and the split (whole
donors held out, no cell in both halves).
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import SGDClassifier

# The label contract, copied from apps/hlca_fedavg/hlca_fedavg/task.py. A FAB
# ships only its own package, so this app cannot import it from there;
# tests/test_contracts.py fails if the two copies drift apart. The order fixes
# the output width, and a client missing a class still returns a coefficient
# matrix of this width.
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
LABEL_COLUMN = "ann_level_2"


# ---------------------------------------------------------------------------
# Step 1. The feature contract
# ---------------------------------------------------------------------------
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


def _check_contracts(a: ad.AnnData, genes: list[str], path: Path) -> None:
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
    if LABEL_COLUMN not in a.obs:
        raise ValueError(f"{path.name}: no obs[{LABEL_COLUMN!r}] column.")
    unknown = sorted(set(a.obs[LABEL_COLUMN].astype(str)) - set(LABELS))
    if unknown:
        raise ValueError(f"{path.name}: labels outside the contract: {unknown}.")
    x = a.X[: min(1000, a.n_obs)]
    values = x.data if hasattr(x, "data") else np.asarray(x).ravel()
    if values.size and (values.min() < 0 or not np.allclose(values, np.rint(values))):
        raise ValueError(
            f"{path.name}: X is not raw counts. The partitions carry counts, and "
            "normalization runs per client."
        )


# ---------------------------------------------------------------------------
# Step 2. The split: whole donors held out
# ---------------------------------------------------------------------------
def split_indices(obs, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole donors for validation.

    Cells from one donor are not independent, so a cell-level split reports a
    number the model cannot repeat on a new patient. Takes the obs table only,
    so baseline.py can size its arrays without loading the matrix.
    """
    if "donor_id" not in obs:
        raise ValueError("No obs['donor_id'] column: the split holds out whole donors.")
    rng = np.random.default_rng(seed)
    donors = obs["donor_id"].astype(str).to_numpy()
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
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("The train and the validation split overlap.")
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Step 3. Load and preprocess ONE client's data
# ---------------------------------------------------------------------------
def preprocess(x) -> np.ndarray:
    """Raw counts -> log2(CPM + 1), as float32.

    The library size is always the row sum over the contract genes, never an
    obs column, so the meaning cannot change when a partition gains one. There
    is no per-client scaling after this: FedAvg needs one shared feature space,
    and each client's own min and max would give every client different units.
    """
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)
    lib = x.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    x /= lib
    x *= 1e6
    np.log2(x + 1, out=x)
    return x


def encode_labels(obs) -> np.ndarray:
    """Cell-type names -> integer codes from LABELS, identical on every node."""
    code = {name: i for i, name in enumerate(LABELS)}
    return obs[LABEL_COLUMN].astype(str).map(code).to_numpy(dtype=np.int64)


def load_partition(path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one partition file: (X_train, y_train, X_val, y_val).

    Checks the contracts before any preprocessing, then normalizes and holds
    out whole donors.
    """
    path = Path(os.path.expanduser(str(path)))
    if not path.is_file():
        raise FileNotFoundError(f"No partition at {path}.")
    genes = load_genes(cfg["gene-list"])
    a = ad.read_h5ad(path)
    _check_contracts(a, genes, path)
    y = encode_labels(a.obs)
    train_idx, val_idx = split_indices(a.obs, float(cfg["val-fraction"]), int(cfg["seed"]))
    x = preprocess(a.X)
    del a
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def client_path(partition_id: int, cfg: dict) -> Path:
    """`<data-dir>/client_<partition_id>.h5ad`. Config, not a glob."""
    data_dir = Path(os.path.expanduser(str(cfg["data-dir"])))
    return data_dir / f"client_{int(partition_id)}.h5ad"


def load_client(partition_id: int, cfg: dict):
    """Load the partition that belongs to this Flower partition id."""
    return load_partition(client_path(partition_id, cfg), cfg)


# ---------------------------------------------------------------------------
# Step 4. The model
# ---------------------------------------------------------------------------
class LogisticRegressionModel:
    """
    The whole model is two arrays:
        coef       (n_labels x n_genes)  one weight per gene, per cell type
        intercept  (n_labels)            one bias per cell type
    Prediction: score_k = coef[k] . x + intercept[k]; predicted type = largest score.

    The weights are float32 to match the float32 data. scikit-learn fits in the
    dtype it was first fitted with, and float64 weights reject float32 input.
    """

    def __init__(self, n_labels, n_genes, lr=0.01):
        self.n_labels = n_labels
        self.n_genes = n_genes
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4,
                                 learning_rate="constant", eta0=lr, random_state=42)

        self.clf.partial_fit(np.zeros((1, n_genes), dtype=np.float32), [0], classes=np.arange(n_labels))
        self.set_weights([np.zeros((n_labels, n_genes)), np.zeros(n_labels)])

    def get_weights(self):
        return [self.clf.coef_.copy(), self.clf.intercept_.copy()]

    def set_weights(self, weights):
        coef = np.array(weights[0], dtype=np.float32)
        intercept = np.array(weights[1], dtype=np.float32)
        if coef.shape != (self.n_labels, self.n_genes) or intercept.shape != (self.n_labels,):
            raise ValueError(
                f"Weights of shape {coef.shape} and {intercept.shape}, the model is "
                f"({self.n_labels}, {self.n_genes}) and ({self.n_labels},)."
            )
        self.clf.coef_ = coef
        self.clf.intercept_ = intercept

    def fit_one_epoch(self, X, y):
        self.clf.partial_fit(X, y, classes=np.arange(self.n_labels))

    def predict(self, X):
        return self.clf.predict(X)


def build_model(cfg: dict) -> LogisticRegressionModel:
    """Fix the shapes from the contracts, not from the data, so every node agrees."""
    return LogisticRegressionModel(len(LABELS), len(load_genes(cfg["gene-list"])), lr=float(cfg["lr"]))


# ---------------------------------------------------------------------------
# Step 5. Train and test
# ---------------------------------------------------------------------------
def train(model, X, y, epochs):
    for _ in range(epochs):
        model.fit_one_epoch(X, y)


def test(model, X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class true positive, false positive and false negative counts.

    Counts, not a score: macro F1 is not linear in the clients, so the server
    sums these and computes one macro F1 from the totals.
    """
    n = len(LABELS)
    pred = model.predict(X) if len(y) else np.zeros(0, dtype=np.int64)
    tp = np.zeros(n, dtype=np.int64)
    fp = np.zeros(n, dtype=np.int64)
    fn = np.zeros(n, dtype=np.int64)
    for c in range(n):
        tp[c] = int(((pred == c) & (y == c)).sum())
        fp[c] = int(((pred == c) & (y != c)).sum())
        fn[c] = int(((pred != c) & (y == c)).sum())
    return tp, fp, fn


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    """Unweighted mean F1 over the classes that appear in the truth.

    A class with no true cell is undefined, not zero, so it is left out rather
    than scored as a failure.
    """
    present = (tp + fn) > 0
    if not present.any():
        return 0.0
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0)
    return float(f1[present].mean())


def per_class_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> list[float | None]:
    """F1 per class in LABELS order, None where the class has no true cell."""
    denom = 2 * tp + fp + fn
    return [None if tp[c] + fn[c] == 0 else float(2 * tp[c] / denom[c]) for c in range(len(tp))]


# ---------------------------------------------------------------------------
# Step 6. Interpretation: top genes per cell type
# ---------------------------------------------------------------------------
def top_genes(coef, genes, labels, k=10):
    """For each cell type, the k genes with the largest positive weight."""
    return {lab: [genes[i] for i in np.argsort(-coef[c])[:k]] for c, lab in enumerate(labels)}
