#!/usr/bin/env python
"""Train the starting model on the two initial training datasets, on one machine.

The two largest HLCA datasets are not held by any institution in the federation,
so a model trained on them leaks nothing from a client. Every arm of the
comparison can start from these weights.

    python apps/hlca_fedavg/train_initial.py --data-dir ~/fedbio-data/hlca

The output loads into the app with:

    flwr run apps/hlca_fedavg --stream -c 'init-model="~/fedbio-data/initial.pt"'

Local, FedAvg and pooled are comparable only when every arm starts from the same
weights. Warm start all three, or none.

The model, the preprocessing and the donor split come from hlca_fedavg.task, so
this script trains exactly what the clients then train.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from hlca_fedavg.task import LABELS, MLP, evaluate, load_partition, macro_f1, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="~/fedbio-data/hlca", help="holds the partition file")
    p.add_argument("--file", default="initial_training.h5ad", help="the file to train on")
    p.add_argument("--gene-list", default=None, help="default: <data-dir>/genes.txt")
    p.add_argument("--out", default="~/fedbio-data/initial.pt", help="where to save the weights")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-size", type=int, default=256, help="must match the app's config")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(os.path.expanduser(args.data_dir))
    gene_list = args.gene_list or str(data_dir / "genes.txt")
    cfg = {
        "gene-list": gene_list,
        "data-dir": str(data_dir),
        "batch-size": args.batch_size,
        "val-fraction": args.val_fraction,
        "seed": args.seed,
    }

    torch.manual_seed(args.seed)
    began = time.time()
    train_loader, val_loader, labels = load_partition(data_dir / args.file, cfg)
    n_genes = train_loader.dataset[0][0].shape[0]
    print(
        f"{args.file}: {len(train_loader.dataset):,} train cells, "
        f"{len(val_loader.dataset):,} held out donors' cells, {n_genes} genes",
        flush=True,
    )

    model = MLP(n_genes, args.hidden_size, len(LABELS))
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, 1, args.lr, args.momentum)[-1]
        val_loss, tp, fp, fn = evaluate(model, val_loader, len(labels))
        print(
            f"epoch {epoch:>2}: train loss {loss:.4f}  val loss {val_loss:.4f}  "
            f"val macro F1 {macro_f1(tp, fp, fn):.4f}",
            flush=True,
        )

    val_loss, tp, fp, fn = evaluate(model, val_loader, len(labels))
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    # The sidecar records what the weights fit, because a state dict carries the
    # shapes but not the gene list or the label order they belong to.
    per_class = {
        label: round(macro_f1(tp[i : i + 1], fp[i : i + 1], fn[i : i + 1]), 4)
        for i, label in enumerate(LABELS)
    }
    meta = {
        "source_file": args.file,
        "gene_list": gene_list,
        "n_genes": int(n_genes),
        "hidden_size": args.hidden_size,
        "labels": list(LABELS),
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n_train_cells": len(train_loader.dataset),
        "n_val_cells": len(val_loader.dataset),
        "val_macro_f1": round(macro_f1(tp, fp, fn), 4),
        "val_f1_per_class": per_class,
        "minutes": round((time.time() - began) / 60, 1),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nSaved {out} and {meta_path.name}")
    print(f"val macro F1 {meta['val_macro_f1']:.4f} in {meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
