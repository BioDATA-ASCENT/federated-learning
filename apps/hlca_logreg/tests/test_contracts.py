"""The contracts must hold, and the checks must reject a broken partition.

The data tests run against a real partition under FEDBIO_DATA (default
~/fedbio-data/hlca), the lowest-numbered client_<k>.h5ad there, and corrupt a
copy of it. There is no synthetic fixture, so they skip when no partition is
present. The contract tests that need no data always run.

Run: uv run pytest apps/hlca_logreg/tests -q
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from hlca_logreg.task import LABELS, build_model, load_genes, load_partition, macro_f1, split_indices, top_genes, train
from hlca_logreg.task import test as count_predictions  # not named test_*, so pytest does not collect it

DATA = Path(os.path.expanduser(os.environ.get("FEDBIO_DATA", "~/fedbio-data/hlca")))
GENES = DATA / "genes.txt"
_found = sorted(DATA.glob("client_*.h5ad"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)) if DATA.is_dir() else []
CLIENT = _found[0] if _found else DATA / "client_0.h5ad"

needs_data = pytest.mark.skipif(
    not (CLIENT.is_file() and GENES.is_file()),
    reason=f"No partition and genes.txt under {DATA}. Set FEDBIO_DATA.",
)


def cfg(**over) -> dict:
    base = {"gene-list": str(GENES), "data-dir": str(DATA), "lr": 0.01, "val-fraction": 0.2, "seed": 42}
    base.update(over)
    return base


def write_variant(tmp_path: Path, change) -> Path:
    """Copy the partition, apply `change` to it, and return the copy's path."""
    a = change(ad.read_h5ad(CLIENT))
    out = tmp_path / CLIENT.name
    a.write_h5ad(out)
    return out


# ---------------------------------------------------------------------------
# Contracts that need no data
# ---------------------------------------------------------------------------
def extract_labels(source: str) -> tuple[str, ...]:
    """The LABELS tuple from a task.py, read without importing it."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "LABELS":
            return tuple(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LABELS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("No LABELS in that file.")


def test_labels_match_the_other_app():
    """Both Track A apps must agree on the label contract."""
    other = Path(__file__).resolve().parents[2] / "hlca_fedavg/hlca_fedavg/task.py"
    if not other.is_file():
        pytest.skip(f"{other} is not in this checkout.")
    assert LABELS == extract_labels(other.read_text())


def test_partition_id_comes_from_the_filename(tmp_path):
    """A node holding only client_4.h5ad is partition 4, not index 0 of a list."""
    from hlca_logreg.client_app import _partition_id

    (tmp_path / "client_4.h5ad").touch()
    ctx = SimpleNamespace(node_config={}, run_config={"data-dir": str(tmp_path)})
    assert _partition_id(ctx) == 4
    ctx.node_config = {"partition-id": 2}
    assert _partition_id(ctx) == 2


def test_baseline_ignores_non_client_files(tmp_path):
    """evaluation.h5ad and initial_training.h5ad must never be treated as clients."""
    import baseline

    for name in ("client_0.h5ad", "client_10.h5ad", "client_2.h5ad", "evaluation.h5ad", "initial_training.h5ad"):
        (tmp_path / name).touch()
    assert [p.name for p in baseline.client_files(tmp_path)] == ["client_0.h5ad", "client_2.h5ad", "client_10.h5ad"]


def test_baseline_budget_comes_from_the_app_config():
    import baseline

    c = baseline.app_config()
    assert int(c["num-server-rounds"]) >= 1 and int(c["local-epochs"]) >= 1


def test_donor_split_keeps_each_donor_on_one_side():
    import pandas as pd

    obs = pd.DataFrame({"donor_id": np.repeat(["a", "b", "c", "d", "e"], [50, 30, 10, 5, 5])})
    tr, va = split_indices(obs, 0.2, 42)
    assert len(tr) + len(va) == len(obs) and len(va) > 0
    assert not set(obs["donor_id"].iloc[tr]) & set(obs["donor_id"].iloc[va])


def test_macro_f1_ignores_absent_classes():
    """A class with no true cell is undefined, not a zero that drags the mean."""
    n = len(LABELS)
    tp, fp, fn = (np.zeros(n, dtype=np.int64) for _ in range(3))
    tp[0] = 10
    assert macro_f1(tp, fp, fn) == 1.0
    fn[1] = 5
    assert macro_f1(tp, fp, fn) == 0.5


# ---------------------------------------------------------------------------
# Contracts checked against a real partition
# ---------------------------------------------------------------------------
@needs_data
def test_good_partition_loads():
    X_tr, y_tr, X_val, y_val = load_partition(CLIENT, cfg())
    assert X_tr.dtype == np.float32
    assert X_tr.shape[1] == len(load_genes(GENES))
    assert len(y_tr) > 0 and len(y_val) > 0
    assert int(max(y_tr.max(), y_val.max())) < len(LABELS)


@needs_data
def test_reordered_genes_are_rejected(tmp_path):
    """The silent failure: widths still agree, so nothing else would notice."""

    def swap(a):
        order = np.arange(a.n_vars)
        order[[0, 1]] = order[[1, 0]]
        return a[:, order].copy()

    with pytest.raises(ValueError, match="gene order differs"):
        load_partition(write_variant(tmp_path, swap), cfg())


@needs_data
def test_missing_gene_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="genes, the contract has"):
        load_partition(write_variant(tmp_path, lambda a: a[:, : a.n_vars - 1].copy()), cfg())


@needs_data
def test_label_outside_the_contract_is_rejected(tmp_path):
    def relabel(a):
        col = a.obs["ann_level_2"].astype(str).copy()
        col.iloc[0] = "Mesothelium"
        a.obs["ann_level_2"] = col
        return a

    with pytest.raises(ValueError, match="labels outside the contract"):
        load_partition(write_variant(tmp_path, relabel), cfg())


@needs_data
def test_normalized_matrix_is_rejected(tmp_path):
    """A partition that was already normalized would be normalized twice."""

    def normalize(a):
        a.X = a.X.multiply(0.5).tocsr()
        return a

    with pytest.raises(ValueError, match="not raw counts"):
        load_partition(write_variant(tmp_path, normalize), cfg())


@needs_data
def test_an_n_counts_column_does_not_change_the_features(tmp_path):
    """The library size is the row sum, whatever obs columns a partition carries."""

    def add_n_counts(a):
        a.obs["n_counts"] = np.full(a.n_obs, 1e7)
        return a

    base = load_partition(CLIENT, cfg())[0]
    changed = load_partition(write_variant(tmp_path, add_n_counts), cfg())[0]
    np.testing.assert_array_equal(base, changed)


@needs_data
def test_split_is_disjoint_by_donor():
    a = ad.read_h5ad(CLIENT, backed="r")
    tr, va = split_indices(a.obs, 0.2, 42)
    assert len(tr) + len(va) == a.n_obs and len(va) > 0
    donors = a.obs["donor_id"].astype(str).to_numpy()
    assert not set(donors[tr]) & set(donors[va])


@needs_data
def test_full_width_model_and_exact_weight_round_trip():
    """A client missing classes still returns all nine rows, and set/get is exact."""
    X_tr, y_tr, X_val, y_val = load_partition(CLIENT, cfg())
    model = build_model(cfg())
    train(model, X_tr, y_tr, 1)
    coef, intercept = model.get_weights()
    assert coef.shape == (len(LABELS), X_tr.shape[1]) and intercept.shape == (len(LABELS),)

    other = build_model(cfg())
    other.set_weights([coef, intercept])
    for a, b in zip(other.get_weights(), (coef, intercept)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(other.predict(X_val), model.predict(X_val))

    tops = top_genes(coef, load_genes(GENES), LABELS, k=10)
    assert list(tops) == list(LABELS) and all(len(v) == 10 for v in tops.values())


@needs_data
def test_counts_sum_to_the_cells_evaluated():
    """Every cell lands in exactly one class as a true positive or a false negative."""
    X_tr, y_tr, X_val, y_val = load_partition(CLIENT, cfg())
    model = build_model(cfg())
    train(model, X_tr, y_tr, 1)
    tp, fp, fn = count_predictions(model, X_val, y_val)
    assert int((tp + fn).sum()) == len(y_val)
    assert int((tp + fp).sum()) == len(y_val)
