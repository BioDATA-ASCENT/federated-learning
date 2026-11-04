"""What the server runs: FedAvg over the connected clients, one per lab group.

The server holds no client data. The gene list comes from genes.txt and the
node count from the run config. Each client evaluates the global model on its
own held-out donors and returns per-class counts; the server sums them and
computes one macro F1 per round.

Outputs, under <out-dir>/<timestamp>/:
    metrics.jsonl            one line per client reply per round, plus the federated line
    final_weights.npz        coef and intercept of the last global model
    top_genes_federated.csv  the top 10 genes per cell type (Stage 3)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from hlca_logreg.task import LABELS, load_genes, macro_f1, per_class_f1, top_genes

app = ServerApp()

COUNT_KEYS = ("tp", "fp", "fn")


class JsonlLogger:
    def __init__(self, out_dir: str, run_config: dict):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(os.path.expanduser(out_dir)) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.t0 = time.time()
        self.write("run_start", config={k: str(v) for k, v in run_config.items()})

    def write(self, kind: str, **fields) -> None:
        rec = {"type": kind, "t": round(time.time() - self.t0, 2), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _drop_partition_id(contents: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Default weighted mean, without the partition label."""
    agg = aggregate_metricrecords(contents, weighting_key)
    return MetricRecord({k: v for k, v in agg.items() if k != "partition-id"})


def _federated_macro_f1(contents: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Sum the per-class counts, then compute one macro F1 from the totals.

    A weighted mean of per-client macro F1 is not the macro F1 of the pooled
    predictions, so the counts are summed first. The weighted means of the
    per-client scores are kept beside it for the table.
    """
    n_classes = len(LABELS)
    totals = {k: np.zeros(n_classes, dtype=np.int64) for k in COUNT_KEYS}
    for content in contents:
        for record in content.metric_records.values():
            for key in COUNT_KEYS:
                totals[key] += np.asarray(record[key], dtype=np.int64)

    scalars = [
        RecordDict({name: MetricRecord({k: v for k, v in record.items() if k not in COUNT_KEYS})})
        for content in contents
        for name, record in content.metric_records.items()
    ]
    agg = _drop_partition_id(scalars, weighting_key)
    agg["macro_f1"] = macro_f1(totals["tp"], totals["fp"], totals["fn"])
    for key in COUNT_KEYS:
        agg[key] = totals[key].tolist()
    return agg


class LoggingFedAvg(FedAvg):
    """Plain FedAvg that records every client reply before aggregating."""

    def __init__(self, logger: JsonlLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def _log(self, kind: str, server_round: int, replies: list[Message]) -> None:
        for msg in replies:
            node = str(msg.metadata.src_node_id)
            if msg.has_error():
                self.logger.write(f"{kind}_error", round=server_round, node=node, error=str(msg.error))
                continue
            for record in msg.content.metric_records.values():
                self.logger.write(kind, round=server_round, node=node, **dict(record))

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self._log("client_train", server_round, replies)
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self._log("client_eval", server_round, replies)
        agg = super().aggregate_evaluate(server_round, replies)
        if agg is not None:
            tp, fp, fn = (np.asarray(agg[k]) for k in COUNT_KEYS)
            per_class = dict(zip(LABELS, per_class_f1(tp, fp, fn)))
            n_nodes = sum(not m.has_error() for m in replies)
            self.logger.write("federated_eval", round=server_round, nodes=n_nodes,
                              per_class_f1=per_class, **dict(agg))
            shown = "  ".join(f"{lab}: {'-' if v is None else f'{v:.3f}'}" for lab, v in per_class.items())
            print(f"[round {server_round}] {n_nodes} nodes, federated macro F1 "
                  f"{agg['macro_f1']:.4f}\n    {shown}", flush=True)
        return agg


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    genes = load_genes(cfg["gene-list"])
    logger = JsonlLogger(str(cfg["out-dir"]), cfg)
    logger.write("labels", labels=list(LABELS))
    print(f"Writing metrics to {logger.path}", flush=True)

    # The starting model: all weights zero, with the full shape. baseline.py
    # starts local and pooled from zero too, so every arm starts alike.
    initial = ArrayRecord([np.zeros((len(LABELS), len(genes)), dtype=np.float32),
                           np.zeros(len(LABELS), dtype=np.float32)])

    min_nodes = int(cfg["min-nodes"])
    strategy = LoggingFedAvg(
        logger,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
        train_metrics_aggr_fn=_drop_partition_id,
        evaluate_metrics_aggr_fn=_federated_macro_f1,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=initial,
        train_config=ConfigRecord({"lr": float(cfg["lr"]), "local-epochs": int(cfg["local-epochs"])}),
        num_rounds=int(cfg["num-server-rounds"]),
    )

    coef, intercept = result.arrays.to_numpy_ndarrays()
    np.savez(logger.dir / "final_weights.npz", coef=coef, intercept=intercept)
    pd.DataFrame(top_genes(coef, genes, LABELS, k=10)).to_csv(logger.dir / "top_genes_federated.csv", index=False)
    logger.write("run_end")
    print(f"Done. Outputs in {logger.dir}", flush=True)
