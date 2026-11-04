#!/usr/bin/env python
"""First split of the HLCA core into initial training, evaluation, and six institutions.

Run from the project folder with the environment active:

    python scripts/partition_hlca.py

Output in data/hlca/:
    initial_training.h5ad             the two largest datasets not in INSTITUTIONS
    evaluation.h5ad                   the other two datasets not in INSTITUTIONS
    client_1.h5ad ... client_6.h5ad   one file per institution, from INSTITUTIONS
    contract.json                     shared gene list, label list, and file summaries
    hvg_genes.json                    the saved gene list, reused by later runs

Every file holds raw counts in X and the same 2000 genes in the same order.

The 2000 genes are chosen from all cells in the two initial training datasets, so no
institution's private cells and no evaluation cells affect them. The result matches
this scanpy call on those cells:

    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="dataset")

The choice is made once and saved in hvg_genes.json. Later runs reuse it as long as the
initial training datasets stay the same. Delete that file to force a new choice.

The source is too large to load into a laptop's memory, so it is read in blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

# ---- Edit these to change the split --------------------------------------------
INSTITUTIONS = {
    "client_1": ["Krasnow_2020"],
    "client_2": ["Meyer_2019", "Teichmann_Meyer_2019"],
    "client_3": ["Jain_Misharin_2021_10Xv1", "Jain_Misharin_2021_10Xv2"],
    "client_4": ["Misharin_Budinger_2018"],
    "client_5": ["Seibold_2020_10Xv2", "Seibold_2020_10Xv3"],
    "client_6": ["Lafyatis_Rojas_2019_10Xv1", "Lafyatis_Rojas_2019_10Xv2"],
}
# Optional: make an institution smaller by keeping a random set of its cells.
# Example: MAX_CELLS = {"client_6": 5000}
MAX_CELLS: dict[str, int] = {}
# ---------------------------------------------------------------------------------

DATASET_COL = "dataset"
LABEL_COL = "ann_level_2"
DONOR_COL = "donor_id"
KEEP_COLS = [DATASET_COL, LABEL_COL, DONOR_COL, "study", "ann_finest_level", "assay"]
# HLCA sets a level to the text "None" when a cell has no annotation at that level.
MISSING_LABELS = {"", "na", "nan", "none", "unknown"}

N_INITIAL = 2
N_EVALUATION = 2
N_GENES = 2000
TARGET_SUM = 1e4
N_BINS = 20
MAX_VALUES_PER_BLOCK = 20_000_000
SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[1]


def say(message: str = "") -> None:
    print(message, flush=True)


def read_blocks(f: h5py.File, codes: np.ndarray, label: str):
    """Yield consecutive blocks of raw counts, skipping blocks with no cells in use."""
    group = f["raw/X"]
    indptr = group["indptr"][:]
    n_cells = len(codes)
    start = 0
    began = time.time()
    next_report = 0.1
    while start < n_cells:
        end = int(np.searchsorted(indptr, indptr[start] + MAX_VALUES_PER_BLOCK, side="right")) - 1
        end = min(max(end, start + 1), n_cells)
        block_codes = codes[start:end]
        if (block_codes >= 0).any():
            a, b = int(indptr[start]), int(indptr[end])
            yield block_codes, indptr[start:end + 1] - a, group["indices"][a:b], group["data"][a:b]
        start = end
        done = start / n_cells
        if done >= next_report or start == n_cells:
            minutes = (time.time() - began) / 60
            say(f"  {label}: {done:4.0%} done, {minutes:.1f} min so far")
            next_report = done + 0.1


def gene_stats(f: h5py.File, batch: np.ndarray, n_batches: int, n_genes: int):
    """Per batch and gene: the sum and squared sum of normalized values, and detections."""
    slots = n_batches * n_genes
    sum_x, sum_x2 = np.zeros(slots), np.zeros(slots)
    n_nonzero = np.zeros(slots, dtype=np.int64)
    for codes, ptr, indices, data in read_blocks(f, batch, "Step 1 of 2, choosing genes"):
        row = np.repeat(np.arange(len(codes)), np.diff(ptr))
        totals = np.bincount(row, weights=data, minlength=len(codes))
        scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0)
        keep = (codes[row] >= 0) & (data != 0)
        row = row[keep]
        slot = codes[row].astype(np.int64) * n_genes + indices[keep]
        normalized = data[keep].astype(np.float64) * scale[row]
        del data, indices, keep
        sum_x += np.bincount(slot, weights=normalized, minlength=slots)
        sum_x2 += np.bincount(slot, weights=normalized * normalized, minlength=slots)
        n_nonzero += np.bincount(slot, minlength=slots)
    shape = (n_batches, n_genes)
    return sum_x.reshape(shape), sum_x2.reshape(shape), n_nonzero.reshape(shape)


def rank_genes_one_batch(n: int, sum_x: np.ndarray, sum_x2: np.ndarray, n_nonzero: np.ndarray):
    """One batch's normalized dispersions and top genes, as in scanpy's flavor='seurat'."""
    expressed = n_nonzero > 0
    mean = sum_x[expressed] / n
    variance = np.maximum((sum_x2[expressed] - n * mean * mean) / (n - 1), 0)
    dispersion = variance / mean
    dispersion[dispersion == 0] = np.nan

    table = pd.DataFrame({"dispersion": np.log(dispersion)})
    table["bin"] = pd.cut(np.log1p(mean), bins=N_BINS).astype(str)
    grouped = table.groupby("bin")["dispersion"]
    average = grouped.transform("mean").to_numpy().copy()
    spread = grouped.transform("std").to_numpy().copy()
    lone = np.isnan(spread)  # a bin with one usable gene gets a normalized dispersion of 1
    spread[lone] = average[lone]
    average[lone] = 0
    norm = (table["dispersion"].to_numpy() - average) / spread

    usable = np.sort(norm[~np.isnan(norm)])[::-1]
    cutoff = usable[min(N_GENES, usable.size) - 1]
    full_norm = np.zeros(len(expressed))
    full_top = np.zeros(len(expressed), dtype=bool)
    full_norm[expressed] = norm
    full_top[expressed] = np.nan_to_num(norm, nan=-np.inf) >= cutoff
    return full_norm, full_top


def choose_genes(f: h5py.File, batch: np.ndarray, n_batches: int, gene_ids: pd.Index) -> np.ndarray:
    """Rank by how many batches pick a gene, then by mean normalized dispersion."""
    sum_x, sum_x2, n_nonzero = gene_stats(f, batch, n_batches, len(gene_ids))
    n_cells = np.bincount(batch[batch >= 0], minlength=n_batches)
    results = [rank_genes_one_batch(int(n_cells[b]), sum_x[b], sum_x2[b], n_nonzero[b]) for b in range(n_batches)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # genes with no usable value
        mean_norm = np.nanmean(np.vstack([r[0] for r in results]), axis=0)
    n_top = np.vstack([r[1] for r in results]).sum(axis=0)
    ranking = pd.DataFrame({"n_top": n_top, "norm": mean_norm}, index=gene_ids).sort_index()
    ranking = ranking.sort_values(["n_top", "norm"], ascending=False, na_position="last")
    return np.sort(gene_ids.get_indexer(ranking.index[:N_GENES]))  # fixed order: as in the source


def load_or_choose_genes(f: h5py.File, path: Path, initial: list[str], dataset: np.ndarray, genes: pd.DataFrame) -> np.ndarray:
    """Reuse the saved gene list if it came from the same datasets. Otherwise make it and save it."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        positions = genes.index.get_indexer(saved["genes"])
        if sorted(saved["initial_training"]) == sorted(initial) and len(positions) == N_GENES and (positions >= 0).all():
            say(f"Using the saved gene list in {path.name}, so step 1 is skipped.")
            return np.sort(positions)
        say(f"{path.name} does not match this split, so choosing the genes again.")

    batch_of = {name: b for b, name in enumerate(initial)}
    batch = pd.Series(dataset).map(batch_of).fillna(-1).astype(np.int32).to_numpy()
    keep_genes = choose_genes(f, batch, len(initial), genes.index)
    path.write_text(json.dumps({
        "initial_training": initial,
        "genes": genes.index[keep_genes].astype(str).tolist(),
        "gene_symbols": genes["feature_name"].iloc[keep_genes].astype(str).tolist() if "feature_name" in genes else None,
    }, indent=1), encoding="utf-8")
    say(f"Saved the gene list in {path.name}.")
    return keep_genes


def collect_counts(f: h5py.File, group: np.ndarray, n_groups: int, n_genes: int, keep_genes: np.ndarray):
    """Gather each group's cells, in source order, on the chosen genes."""
    parts: list[list] = [[] for _ in range(n_groups)]
    for codes, ptr, indices, data in read_blocks(f, group, "Step 2 of 2, collecting cells"):
        block = sparse.csr_matrix((data, indices, ptr), shape=(len(codes), n_genes))[:, keep_genes]
        block.eliminate_zeros()
        block.sort_indices()
        for g in np.unique(codes[codes >= 0]):
            parts[g].append(block[codes == g])
    merged = []
    for g in range(n_groups):
        merged.append(sparse.vstack(parts[g], format="csr", dtype=np.float32))
        parts[g] = None  # free each group's pieces as soon as they are joined
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the HLCA core for the federated simulation.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw" / "hlca_core.h5ad")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "hlca")
    args = parser.parse_args()
    if not args.input.exists():
        sys.exit(f"Cannot find {args.input}. Download it first (see quickstart.md).")
    args.output.mkdir(parents=True, exist_ok=True)

    began = time.time()
    with h5py.File(args.input, "r") as f:
        if f["raw/X"].attrs.get("encoding-type") != "csr_matrix":
            sys.exit("Expected raw/X to be stored by cell (csr_matrix), so this script cannot read this file.")
        obs = read_elem(f["obs"])
        obs = obs[[c for c in KEEP_COLS if c in obs]]
        genes = read_elem(f["raw/var"])
        dataset = obs[DATASET_COL].astype(str).to_numpy()

        text = obs[LABEL_COL].astype(object).astype(str).str.strip().str.lower()
        labelled = (obs[LABEL_COL].notna() & ~text.isin(MISSING_LABELS)).to_numpy()
        sizes = pd.Series(dataset[labelled]).value_counts()

        # Assign each dataset to a group. A misspelled name would otherwise lose its cells.
        taken = [name for names in INSTITUTIONS.values() for name in names]
        unknown = sorted(set(taken) - set(sizes.index))
        if unknown:
            sys.exit(f"These names in INSTITUTIONS are not in the file: {unknown}\nThe file has: {sorted(sizes.index)}")
        leftover = sizes.drop(taken)
        if len(leftover) != N_INITIAL + N_EVALUATION:
            sys.exit(f"Expected {N_INITIAL + N_EVALUATION} datasets outside INSTITUTIONS, found {list(leftover.index)}.")
        initial = list(leftover.index[:N_INITIAL])
        evaluation = list(leftover.index[N_INITIAL:])
        if sizes.drop(initial).max() > sizes[initial].min():
            say(f"NOTE: {sizes.drop(initial).idxmax()} is larger than an initial training dataset.")

        plan = {"initial_training": initial, "evaluation": evaluation, **INSTITUTIONS}
        group_names = list(plan)
        group_of = {name: g for g, names in enumerate(plan.values()) for name in names}
        group = pd.Series(dataset).map(group_of).fillna(-1).astype(np.int32).to_numpy()
        group[~labelled] = -1

        rng = np.random.default_rng(SEED)
        for name, limit in MAX_CELLS.items():
            rows = np.flatnonzero(group == group_names.index(name))
            if len(rows) > limit:
                group[rng.choice(rows, len(rows) - limit, replace=False)] = -1

        say(f"{'file':<20} {'cells':>8} {'donors':>7}  datasets")
        donor_sets = {}
        for g, name in enumerate(group_names):
            donors = set(obs.loc[group == g, DONOR_COL].astype(str)) if DONOR_COL in obs else set()
            donor_sets[name] = donors
            say(f"{name:<20} {(group == g).sum():>8,} {len(donors):>7,}  {', '.join(plan[name])}")
        say(f"Cells removed: {(group < 0).sum():,}\n")
        for name, donors in donor_sets.items():
            if name != "evaluation" and donors & donor_sets["evaluation"]:
                say(f"WARNING: evaluation shares {len(donors & donor_sets['evaluation'])} donor(s) with {name}.")

        keep_genes = load_or_choose_genes(f, args.output / "hvg_genes.json", initial, dataset, genes)
        counts = collect_counts(f, group, len(group_names), len(genes), keep_genes)

    labels = sorted(obs.loc[group >= 0, LABEL_COL].astype(str).unique())
    say(f"\n{len(labels)} labels from {LABEL_COL}: {', '.join(labels)}")
    var = pd.DataFrame(index=genes.index[keep_genes].astype(str))
    if "feature_name" in genes:
        var["feature_name"] = genes["feature_name"].iloc[keep_genes].astype(str).to_numpy()

    say("\nWriting the files ...")
    files = []
    for g, name in enumerate(group_names):
        cells = obs.iloc[np.flatnonzero(group == g)].copy()
        for column in cells.columns:
            if column == LABEL_COL:
                cells[column] = pd.Categorical(cells[column].astype(str), categories=labels)
            elif isinstance(cells[column].dtype, pd.CategoricalDtype):
                cells[column] = cells[column].cat.remove_unused_categories()
        path = args.output / f"{name}.h5ad"
        ad.AnnData(X=counts[g], obs=cells, var=var.copy(), uns={"group": name}).write_h5ad(path, compression="gzip")
        counts[g] = None
        present = cells[LABEL_COL].value_counts()
        files.append({
            "group": name,
            "file": path.name,
            "datasets": plan[name],
            "n_cells": len(cells),
            "n_donors": len(donor_sets[name]),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "labels_absent": [k for k in labels if present.get(k, 0) == 0],
        })
        say(f"  {path.name}: {len(cells):,} cells, {files[-1]['size_mb']:,.0f} MB")

    contract = {
        "source": args.input.name,
        "label_column": LABEL_COL,
        "gene_selection": f"{N_GENES} genes from all cells in the initial_training datasets, batch_key='{DATASET_COL}'",
        "max_cells": MAX_CELLS,
        "files": files,
        "labels": labels,
        "genes": var.index.tolist(),
        "gene_symbols": var["feature_name"].tolist() if "feature_name" in var else None,
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    say(f"\nDone in {(time.time() - began) / 60:.1f} min. Files are in {args.output}")


if __name__ == "__main__":
    main()