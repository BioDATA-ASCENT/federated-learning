"""ServerApp: runs on the SuperLink laptop.

FedAvg over the connected SuperNodes. Every round it writes one JSON line per
event to <out-dir>/<run-timestamp>/metrics.jsonl, which dashboard.py plots live.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from mnist_fedavg.task import MLP, load_test_data, test

app = ServerApp()


class JsonlLogger:
    def __init__(self, out_dir: str, run_config: dict):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(os.path.expanduser(out_dir)) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.t0 = time.time()
        self.write("run_start", config=run_config)

    def write(self, kind: str, **fields) -> None:
        rec = {"type": kind, "t": round(time.time() - self.t0, 2), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _drop_partition_id(contents: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Default weighted mean, minus the partition-id label (averaging it is meaningless)."""
    agg = aggregate_metricrecords(contents, weighting_key)
    return MetricRecord({k: v for k, v in agg.items() if k != "partition-id"})


class LoggingFedAvg(FedAvg):
    """Plain FedAvg that also records every client's reply before aggregating."""

    def __init__(self, logger: JsonlLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def _log_replies(self, kind: str, server_round: int, replies: list[Message]) -> None:
        for msg in replies:
            node = str(msg.metadata.src_node_id)
            if msg.has_error():
                self.logger.write(f"{kind}_error", round=server_round, node=node, error=str(msg.error))
                continue
            metrics = dict(msg.content["metrics"])
            self.logger.write(kind, round=server_round, node=node, **metrics)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self._log_replies("client_train", server_round, replies)
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self._log_replies("client_eval", server_round, replies)
        agg = super().aggregate_evaluate(server_round, replies)
        if agg is not None:
            self.logger.write("fed_eval", round=server_round, **dict(agg))
        return agg


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    logger = JsonlLogger(cfg["out-dir"], cfg)
    print(f"Writing metrics to {logger.path}", flush=True)

    torch.manual_seed(int(cfg["seed"]))
    initial = ArrayRecord(MLP().state_dict())
    test_loader = load_test_data(cfg["data-root"])

    def central_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = MLP()
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, test_loader)
        logger.write("global_eval", round=server_round, test_loss=loss, test_acc=acc)
        return MetricRecord({"test_loss": loss, "test_acc": acc})

    min_nodes = int(cfg["min-nodes"])
    strategy = LoggingFedAvg(
        logger,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
        train_metrics_aggr_fn=_drop_partition_id,
        evaluate_metrics_aggr_fn=_drop_partition_id,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=initial,
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=central_eval,
    )

    torch.save(result.arrays.to_torch_state_dict(), logger.dir / "final_model.pt")
    logger.write("run_end")
    print(f"Done. Metrics: {logger.path}  Model: {logger.dir / 'final_model.pt'}", flush=True)
