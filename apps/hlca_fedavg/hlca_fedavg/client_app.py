"""ClientApp: runs on every SuperNode, on that node's own partition.

In simulation the partition id comes from the federation. In deployment a
SuperNode can be started with
    --node-config 'partition-id=<0..N-1> num-partitions=<N>'
Without it, the node uses the one client_<k>.h5ad file in its data-dir, so a
laptop that holds one institution's file joins as that institution and needs no
ID (scripts/join_federation.sh passes none).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hlca_fedavg.task import (
    build_model,
    evaluate_detailed,
    gene_list_hash,
    load_client,
    load_genes,
    macro_f1,
    train,
)

app = ClientApp()


def _partition_id(context: Context) -> int:
    """This node's partition: from the node config if set, else the one file it holds."""
    if "partition-id" in context.node_config:
        return int(context.node_config["partition-id"])
    data_dir = Path(os.path.expanduser(str(context.run_config["data-dir"])))
    found = sorted(data_dir.glob("client_*.h5ad"))
    if len(found) != 1:
        raise ValueError(
            f"Cannot tell which partition this node holds: found {len(found)} client_*.h5ad "
            f"files in {data_dir}. Keep exactly one there, or start the node with "
            "--node-config 'partition-id=<k>'."
        )
    return int(found[0].stem.removeprefix("client_"))


def _setup(msg: Message, context: Context):
    pid = _partition_id(context)
    cfg = dict(context.run_config)
    torch.manual_seed(int(cfg["seed"]) + pid)
    model = build_model(cfg)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_loader, val_loader, labels = load_client(pid, cfg)
    return pid, cfg, model, train_loader, val_loader, labels


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    pid, cfg, model, train_loader, _, _ = _setup(msg, context)
    rnd = msg.content["config"]["server-round"]
    losses = train(
        model,
        train_loader,
        int(cfg["local-epochs"]),
        float(msg.content["config"]["lr"]),
        float(cfg["momentum"]),
    )
    n = len(train_loader.dataset)
    print(f"[client {pid}] round {rnd}: {n} cells, loss {losses[-1]:.4f}", flush=True)
    metrics = MetricRecord({"num-examples": n, "train_loss": losses[-1], "partition-id": pid})
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.query()
def query_fn(msg: Message, context: Context) -> Message:
    """Readiness check (run with check-only=true): load and check the partition, train nothing.

    Uses the same loader and contract checks as a training round. A failure is
    reported as the reason, not raised, so the server can print it.
    """
    cfg = dict(context.run_config)
    try:
        pid = _partition_id(context)
        train_loader, val_loader, _ = load_client(pid, cfg)
        status = {
            "ready": 1,
            "partition-id": pid,
            "train-cells": len(train_loader.dataset),
            "val-cells": len(val_loader.dataset),
            "genes-hash": gene_list_hash(load_genes(cfg["gene-list"])),
            "detail": "ok",
        }
        print(f"[check] ready as client {pid}: {status['train-cells']} train, {status['val-cells']} val cells", flush=True)
    except Exception as e:  # noqa: BLE001  the reason goes back to the server
        status = {"ready": 0, "partition-id": -1, "train-cells": 0, "val-cells": 0,
                  "genes-hash": "", "detail": f"{type(e).__name__}: {e}"}
        print(f"[check] not ready: {status['detail']}", flush=True)
    return Message(content=RecordDict({"status": ConfigRecord(status)}), reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    pid, cfg, model, _, val_loader, labels = _setup(msg, context)
    details = evaluate_detailed(
        model,
        val_loader,
        len(labels),
        max_embedding_cells=int(cfg["umap-sample-per-site"]),
        seed=int(cfg["seed"]) + pid,
    )
    n = len(val_loader.dataset)
    local = macro_f1(details.tp, details.fp, details.fn)
    print(f"[client {pid}] local macro F1 {local:.4f} on {n} cells", flush=True)
    # Confusion counts cover every held-out cell; sampled hidden embeddings and
    # predictions support a shared UMAP on the server.
    metrics = MetricRecord(
        {
            "num-examples": n,
            "eval_loss": details.loss,
            "local_macro_f1": local,
            "partition-id": pid,
            "tp": details.tp.tolist(),
            "fp": details.fp.tolist(),
            "fn": details.fn.tolist(),
            "confusion": details.confusion.ravel().tolist(),
        }
    )
    visualization = ArrayRecord.from_numpy_ndarrays(
        [details.embeddings, details.y_true, details.y_pred]
    )
    return Message(
        content=RecordDict({"metrics": metrics, "visualization": visualization}),
        reply_to=msg,
    )
