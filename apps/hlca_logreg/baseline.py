"""The main results table: local vs FedAvg vs pooled, per client.

Every arm starts from zero weights and gets the same budget: local and pooled
train for num-server-rounds * local-epochs epochs. Those, and lr, seed and
val-fraction, are read from the run's own metrics.jsonl, which records the
config the run actually used, overrides included. Without it they come from
this app's pyproject.toml. Nothing is restated here, so nothing can drift.

Memory: only the pooled training matrix (float32) and one client at a time are
resident. The pooled matrix is sized from the obs tables first and filled in
place, so there is no second concatenated copy.

Run, after a Flower run has written final_weights.npz:
    uv run python apps/hlca_logreg/baseline.py                  # newest run under out-dir
    uv run python apps/hlca_logreg/baseline.py --weights <run>/final_weights.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hlca_logreg.task import (LABELS, build_model, load_genes, load_partition, macro_f1,  # noqa: E402
                              split_indices, test, train)

CLIENT_FILE = re.compile(r"^client_(\d+)\.h5ad$")


def app_config() -> dict:
    with (HERE / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["flwr"]["app"]["config"]


def client_files(data_dir: Path) -> list[Path]:
    """client_<k>.h5ad only, by k. evaluation.h5ad and initial_training.h5ad are not clients."""
    found = [p for p in data_dir.iterdir() if CLIENT_FILE.match(p.name)]
    return sorted(found, key=lambda p: int(CLIENT_FILE.match(p.name).group(1)))


RUN_KEYS = ("num-server-rounds", "local-epochs", "lr", "seed", "val-fraction")


def run_config(weights: Path) -> dict:
    """The config the Flower run used, from the run_start line of its metrics.jsonl."""
    log = weights.parent / "metrics.jsonl"
    if not log.is_file():
        return {}
    with log.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "run_start":
                return {k: rec["config"][k] for k in RUN_KEYS if k in rec["config"]}
    return {}


def newest_weights(out_dir: Path) -> Path:
    runs = sorted(out_dir.glob("*/final_weights.npz"))
    if not runs:
        raise FileNotFoundError(f"No final_weights.npz under {out_dir}. Run the federation first, or pass --weights.")
    return runs[-1]


def main() -> None:
    cfg = app_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, help="final_weights.npz from a Flower run")
    parser.add_argument("--data-dir", default=cfg["data-dir"])
    parser.add_argument("--gene-list", default=cfg["gene-list"])
    args = parser.parse_args()
    weights = args.weights or newest_weights(Path(os.path.expanduser(str(cfg["out-dir"]))))
    from_run = run_config(weights)
    if not from_run:
        print(f"No metrics.jsonl beside {weights}: taking the budget from pyproject.toml.")
    cfg = {**cfg, **from_run, "data-dir": args.data_dir, "gene-list": args.gene_list}

    rounds, epochs = int(cfg["num-server-rounds"]), int(cfg["local-epochs"])
    total_epochs = rounds * epochs  # local and pooled get the same budget as FedAvg
    data_dir = Path(os.path.expanduser(str(cfg["data-dir"])))
    files = client_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No client_<k>.h5ad in {data_dir}.")
    n_genes = len(load_genes(cfg["gene-list"]))
    print(f"{len(files)} clients, {n_genes} genes, {rounds} rounds x {epochs} epoch(s) = "
          f"{total_epochs} epochs per arm, lr {cfg['lr']}, FedAvg weights {weights}")

    # FedAvg: the final global model from the Flower run.
    fed = build_model(cfg)
    w = np.load(weights)
    fed.set_weights([w["coef"], w["intercept"]])

    # Pooled: one model on every client's training cells together. Size first,
    # from obs only, then fill one client at a time.
    sizes = []
    for p in files:
        obs = ad.read_h5ad(p, backed="r").obs
        sizes.append(len(split_indices(obs, float(cfg["val-fraction"]), int(cfg["seed"]))[0]))
    X_pool = np.empty((sum(sizes), n_genes), dtype=np.float32)
    y_pool = np.empty(sum(sizes), dtype=np.int64)
    at = 0
    for p, n in zip(files, sizes):
        X_tr, y_tr, _, _ = load_partition(p, cfg)
        X_pool[at:at + n], y_pool[at:at + n] = X_tr, y_tr
        at += n
        del X_tr, y_tr
    pooled = build_model(cfg)
    train(pooled, X_pool, y_pool, total_epochs)
    del X_pool, y_pool

    # Local, and the scores: one client resident at a time.
    arms = ("local", "fedavg", "pooled")
    totals = {a: [np.zeros(len(LABELS), dtype=np.int64) for _ in range(3)] for a in arms}
    rows = []
    for p in files:
        X_tr, y_tr, X_val, y_val = load_partition(p, cfg)
        local = build_model(cfg)
        train(local, X_tr, y_tr, total_epochs)
        row = {"client": p.stem, "n_train": len(y_tr), "n_val": len(y_val)}
        for arm, model in zip(arms, (local, fed, pooled)):
            counts = test(model, X_val, y_val)
            for t, c in zip(totals[arm], counts):
                t += c
            row[arm] = macro_f1(*counts)
        rows.append(row)
        del X_tr, y_tr, X_val, y_val, local

    # The federation-wide row: counts summed over clients, one macro F1 per arm,
    # which is the number the Flower run reports as macro_f1.
    rows.append({"client": "all (summed counts)", "n_train": sum(r["n_train"] for r in rows),
                 "n_val": sum(r["n_val"] for r in rows),
                 **{arm: macro_f1(*totals[arm]) for arm in arms}})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))
    out = weights.parent / "results_table.csv"
    table.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
