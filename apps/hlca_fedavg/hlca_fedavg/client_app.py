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
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hlca_fedavg.task import build_model, evaluate, load_client, macro_f1, train

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


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    pid, _, model, _, val_loader, labels = _setup(msg, context)
    loss, tp, fp, fn = evaluate(model, val_loader, len(labels))
    n = len(val_loader.dataset)
    local = macro_f1(tp, fp, fn)
    print(f"[client {pid}] local macro F1 {local:.4f} on {n} cells", flush=True)
    # The counts travel, not the score: the server needs the totals to compute
    # one macro F1. local_macro_f1 rides along for the per-client table only.
    metrics = MetricRecord(
        {
            "num-examples": n,
            "eval_loss": loss,
            "local_macro_f1": local,
            "partition-id": pid,
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
