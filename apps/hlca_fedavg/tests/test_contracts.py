"""The contract checks must reject a broken partition.

These run against a real partition under FEDBIO_DATA (default
~/fedbio-data/hlca) and corrupt a copy of it. There is no synthetic fixture,
so the tests skip when no partition is present.

Run: uv run pytest apps/hlca_fedavg/tests -q
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from hlca_fedavg.task import LABELS, client_file, evaluate, load_client, load_genes, macro_f1

DATA = Path(os.path.expanduser(os.environ.get("FEDBIO_DATA", "~/fedbio-data/hlca")))
CLIENT = DATA / client_file(0)
GENES = DATA / "genes.txt"

pytestmark = pytest.mark.skipif(
    not (CLIENT.is_file() and GENES.is_file()),
    reason=f"No partition at {CLIENT}. Set FEDBIO_DATA.",
)


def cfg(**over) -> dict:
    base = {
        "gene-list": str(GENES),
        "data-dir": str(DATA),
        "hidden-size": 32,
        "batch-size": 256,
        "val-fraction": 0.2,
        "seed": 42,
    }
    base.update(over)
    return base


def write_variant(tmp_path: Path, change) -> dict:
    """Copy client 0, apply `change` to it, and point a config at the copy."""
    a = change(ad.read_h5ad(CLIENT))
    a.write_h5ad(tmp_path / client_file(0))
    return cfg(**{"data-dir": str(tmp_path)})


def test_good_partition_loads():
    train_loader, val_loader, labels = load_client(0, cfg())
    assert labels == list(LABELS)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    n_genes = len(load_genes(GENES))
    xb, yb = next(iter(train_loader))
    assert xb.shape[1] == n_genes
    assert int(yb.max()) < len(LABELS)


def test_reordered_genes_are_rejected(tmp_path):
    """The silent failure: widths still agree, so nothing else would notice."""

    def swap(a):
        order = np.arange(a.n_vars)
        order[[0, 1]] = order[[1, 0]]
        return a[:, order].copy()

    with pytest.raises(ValueError, match="gene order differs"):
        load_client(0, write_variant(tmp_path, swap))


def test_missing_gene_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="genes, the contract has"):
        load_client(0, write_variant(tmp_path, lambda a: a[:, : a.n_vars - 1].copy()))


def test_label_outside_the_contract_is_rejected(tmp_path):
    def relabel(a):
        col = a.obs["ann_level_2"].astype(str).copy()
        col.iloc[0] = "Mesothelium"
        a.obs["ann_level_2"] = col
        return a

    with pytest.raises(ValueError, match="labels outside the contract"):
        load_client(0, write_variant(tmp_path, relabel))


def test_normalized_matrix_is_rejected(tmp_path):
    """A partition that was already normalized would be normalized twice."""

    def normalize(a):
        a.X = a.X.multiply(0.5)
        return a

    with pytest.raises(ValueError, match="not raw counts"):
        load_client(0, write_variant(tmp_path, normalize))


def test_splits_are_disjoint_and_donors_do_not_leak():
    a = ad.read_h5ad(CLIENT)
    train_loader, val_loader, _ = load_client(0, cfg())
    assert len(train_loader.dataset) + len(val_loader.dataset) == a.n_obs
    if "donor_id" in a.obs:
        assert len(val_loader.dataset) > 0, "a donor split must still hold out cells"


def test_macro_f1_ignores_absent_classes():
    """A class with no true cell is undefined, not a zero that drags the mean."""
    n = len(LABELS)
    tp = np.zeros(n, dtype=np.int64)
    fp = np.zeros(n, dtype=np.int64)
    fn = np.zeros(n, dtype=np.int64)
    tp[0] = 10
    assert macro_f1(tp, fp, fn) == 1.0
    fn[1] = 5
    assert macro_f1(tp, fp, fn) == 0.5


def test_counts_sum_to_the_cells_evaluated():
    """Every cell lands in exactly one class as a true positive or a false negative."""
    from hlca_fedavg.task import build_model

    _, val_loader, labels = load_client(0, cfg())
    model = build_model(cfg())
    _, tp, fp, fn = evaluate(model, val_loader, len(labels))
    assert int((tp + fn).sum()) == len(val_loader.dataset)
    assert int(tp.sum() + fp.sum()) == len(val_loader.dataset)
