"""What each lab (client) runs: a train handler and an evaluate handler.

In simulation the partition id comes from the federation. In deployment a
SuperNode can be started with
    --node-config 'partition-id=<0..N-1> num-partitions=<N>'
Without it, the node uses the one client_<k>.h5ad file in its data-dir and
takes k from the filename, so a laptop that holds one institution's file joins
as that institution.
"""

from __future__ import annotations

import os
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from hlca_logreg.task import build_model, load_client, macro_f1, test, train

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


def _model(msg: Message, cfg: dict, lr: float | None = None):
    """The global model the server sent, shaped from the contracts, not the data."""
    if lr is not None:
        cfg = {**cfg, "lr": lr}
    model = build_model(cfg)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    return model


@app.train()
def client_train(msg: Message, context: Context):
    pid = _partition_id(context)
    cfg = dict(context.run_config)
    X_train, y_train, _, _ = load_client(pid, cfg)

    # The server sends lr and local-epochs from its run config each round.
    model = _model(msg, cfg, lr=float(msg.content["config"]["lr"]))
    epochs = int(msg.content["config"]["local-epochs"])
    train(model, X_train, y_train, epochs)
    print(f"[client {pid}] round {msg.content['config']['server-round']}: "
          f"{len(y_train)} cells, {epochs} epoch(s)", flush=True)

    # "num-examples" is what FedAvg uses to weight this client.
    content = RecordDict({
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": MetricRecord({"num-examples": len(y_train), "partition-id": pid}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def client_evaluate(msg: Message, context: Context):
    pid = _partition_id(context)
    cfg = dict(context.run_config)
    _, _, X_val, y_val = load_client(pid, cfg)
    model = _model(msg, cfg)

    tp, fp, fn = test(model, X_val, y_val)
    local = macro_f1(tp, fp, fn)
    acc = float(tp.sum() / max(len(y_val), 1))
    print(f"[client {pid}] local macro F1 {local:.4f}  accuracy {acc:.4f}  (n_val = {len(y_val)})", flush=True)
    # The counts travel, not the score: the server sums them and computes one
    # macro F1. local_macro_f1 rides along for the per-client table only.
    content = RecordDict({
        "metrics": MetricRecord({
            "num-examples": len(y_val),
            "local_macro_f1": local,
            "accuracy": acc,
            "partition-id": pid,
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
        }),
    })
    return Message(content=content, reply_to=msg)
