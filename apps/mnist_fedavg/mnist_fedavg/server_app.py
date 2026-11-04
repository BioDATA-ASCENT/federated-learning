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
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
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
    """FedAvg that assigns data partitions to nodes and logs every client's reply.

    Partitions: the first time messages go out, the nodes online at that moment
    are frozen as this run's participants and numbered 0..N-1 (sorted by node
    ID), with N = their count. Every message carries that node's partition-id
    and num-partitions in its config, so nobody passes IDs by hand. Nodes that
    connect later are left out until the next run, which keeps the split fixed.
    """

    def __init__(self, logger: JsonlLogger, response_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.response_timeout = response_timeout
        self.partitions: dict[int, int] = {}  # node_id -> partition-id
        self._configured_nodes: dict[str, set[int]] = {}

    def _remember_configured_nodes(self, phase: str, messages: list[Message]) -> None:
        self._configured_nodes[phase] = {
            msg.metadata.dst_node_id
            for msg in messages
            if msg.metadata.dst_node_id is not None
        }

    def _log_missing_nodes(
        self, phase: str, server_round: int, replies: list[Message]
    ) -> None:
        configured = self._configured_nodes.pop(phase, set())
        replied = {
            msg.metadata.src_node_id
            for msg in replies
            if msg.metadata.src_node_id is not None
        }
        for node in sorted(configured - replied):
            self.logger.write(
                "node_timeout",
                round=server_round,
                phase=phase,
                node=str(node),
                response_timeout_s=self.response_timeout,
            )
            print(
                f"[server] node {node} timed out during {phase} in round "
                f"{server_round}; skipping it for this phase",
                flush=True,
            )

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> list[Message]:
        if not self.partitions:
            self.partitions = {nid: i for i, nid in enumerate(sorted(node_ids))}
            n = len(self.partitions)
            assignment = ", ".join(
                f"node {nid} -> {partition}"
                for nid, partition in self.partitions.items()
            )
            print(f"Assigned {n} partitions: {assignment}", flush=True)
            self.logger.write(
                "partitions",
                num_partitions=n,
                assignment={str(k): v for k, v in self.partitions.items()},
            )

        late = [nid for nid in node_ids if nid not in self.partitions]
        if late:
            print(
                f"Skipping {len(late)} node(s) that joined mid-run "
                f"(they join from the next run): {late}",
                flush=True,
            )

        messages = []
        for nid in node_ids:
            if nid not in self.partitions:
                continue
            config = ConfigRecord(dict(record[self.configrecord_key]))
            config["partition-id"] = self.partitions[nid]
            config["num-partitions"] = len(self.partitions)
            content = RecordDict(
                {
                    self.arrayrecord_key: record[self.arrayrecord_key],
                    self.configrecord_key: config,
                }
            )
            messages.append(
                Message(
                    content=content,
                    message_type=message_type,
                    dst_node_id=nid,
                )
            )
        return messages

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self._remember_configured_nodes("train", messages)
        return messages

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
        self._log_missing_nodes("train", server_round, replies)
        self._log_replies("client_train", server_round, replies)
        return super().aggregate_train(server_round, replies)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        messages = list(
            super().configure_evaluate(server_round, arrays, config, grid)
        )
        self._remember_configured_nodes("evaluate", messages)
        return messages

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self._log_missing_nodes("evaluate", server_round, replies)
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
        response_timeout=float(cfg["response-timeout-seconds"]),
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
        timeout=float(cfg["response-timeout-seconds"]),
        evaluate_fn=central_eval,
    )

    torch.save(result.arrays.to_torch_state_dict(), logger.dir / "final_model.pt")
    logger.write("run_end")
    print(f"Done. Metrics: {logger.path}  Model: {logger.dir / 'final_model.pt'}", flush=True)
