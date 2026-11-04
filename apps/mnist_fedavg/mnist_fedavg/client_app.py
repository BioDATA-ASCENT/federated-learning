"""ClientApp: runs on every SuperNode, on that laptop's own MNIST share.

The server assigns each node its partition-id and num-partitions and sends them
in every message's config. A SuperNode started with
    --node-config 'partition-id=<i> num-partitions=<N>'
uses those values only when the message carries none (e.g. in simulation).
"""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from mnist_fedavg.task import MLP, load_client_data, test, train

app = ClientApp()


def _setup(msg: Message, context: Context):
    msg_cfg = msg.content["config"]
    if "partition-id" in msg_cfg:  # assigned by the server
        pid, nparts = int(msg_cfg["partition-id"]), int(msg_cfg["num-partitions"])
    else:  # simulation, or a node started with --node-config
        pid = int(context.node_config["partition-id"])
        nparts = int(context.node_config["num-partitions"])
    cfg = dict(context.run_config)
    torch.manual_seed(int(cfg["seed"]) + pid)
    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_loader, val_loader, digits = load_client_data(pid, nparts, cfg)
    return pid, cfg, model, train_loader, val_loader, digits


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    pid, cfg, model, train_loader, _, digits = _setup(msg, context)
    rnd = msg.content["config"].get("server-round", "?")
    t0 = time.time()
    losses = train(model, train_loader, int(cfg["local-epochs"]), float(cfg["lr"]), float(cfg["momentum"]))
    secs = time.time() - t0
    n = len(train_loader.dataset)
    print(
        f"[client {pid}] round {rnd}: trained on {n} samples (digits {digits}), "
        f"epoch losses {[round(l, 4) for l in losses]}, {secs:.1f}s",
        flush=True,
    )
    metrics = MetricRecord(
        {
            "num-examples": n,
            "train_loss": losses[-1],
            "train_time_s": secs,
            "partition-id": pid,
        }
    )
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    pid, _, model, _, val_loader, _ = _setup(msg, context)
    loss, acc = test(model, val_loader)
    print(f"[client {pid}] local val: loss {loss:.4f}, acc {acc:.4f}", flush=True)
    metrics = MetricRecord(
        {
            "num-examples": len(val_loader.dataset),
            "eval_loss": loss,
            "eval_acc": acc,
            "partition-id": pid,
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
