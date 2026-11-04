"""ClientApp: runs on every SuperNode, on that node's own partition.

In simulation the partition id comes from the federation. In deployment each
SuperNode is started with
    --node-config 'partition-id=<0..N-1> num-partitions=<N>'
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hlca_fedavg.task import (
    build_model,
    evaluate_detailed,
    load_client,
    macro_f1,
    train,
)

app = ClientApp()


def _setup(msg: Message, context: Context):
    pid = int(context.node_config["partition-id"])
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
