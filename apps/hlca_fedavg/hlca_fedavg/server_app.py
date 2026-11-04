"""ServerApp: FedAvg over the connected clients, one client per lab group.

There is no central test set, by design: each client evaluates the global model
on its own held-out donors. The server writes one JSON line per client reply
per round to <out-dir>/<timestamp>/metrics.jsonl, which is the per-client
result table.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from umap import UMAP

from hlca_fedavg.task import (
    LABELS,
    build_model,
    evaluate_detailed,
    load_evaluation,
    macro_f1,
)

app = ServerApp()

COUNT_KEYS = ("tp", "fp", "fn")
CONFUSION_KEY = "confusion"


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

    def write_umap(self, server_round: int, points: list[dict]) -> None:
        path = self.dir / "umap"
        path.mkdir(exist_ok=True)
        with (path / f"round-{server_round:04d}.json").open("w") as f:
            json.dump({"round": server_round, "points": points}, f)


def _drop_partition_id(contents: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Default weighted mean, without the partition label."""
    agg = aggregate_metricrecords(contents, weighting_key)
    return MetricRecord({k: v for k, v in agg.items() if k != "partition-id"})


def _federated_macro_f1(contents: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Sum the per-class counts, then compute one macro F1 from the totals.

    A weighted mean of per-client macro F1 is not the macro F1 of the pooled
    predictions, so the counts are summed first. The weighted means of loss and
    of the per-client score are kept beside it for the table.
    """
    n_classes = len(LABELS)
    totals = {k: np.zeros(n_classes, dtype=np.int64) for k in COUNT_KEYS}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for content in contents:
        for record in content.metric_records.values():
            for key in COUNT_KEYS:
                totals[key] += np.asarray(record[key], dtype=np.int64)
            confusion += np.asarray(record[CONFUSION_KEY], dtype=np.int64).reshape(
                n_classes, n_classes
            )

    scalars = [
        RecordDict(
            {
                name: MetricRecord(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in (*COUNT_KEYS, CONFUSION_KEY)
                    }
                )
            }
        )
        for content in contents
        for name, record in content.metric_records.items()
    ]
    agg = _drop_partition_id(scalars, weighting_key)
    agg["macro_f1"] = macro_f1(totals["tp"], totals["fp"], totals["fn"])
    for key in COUNT_KEYS:
        agg[key] = totals[key].tolist()
    agg[CONFUSION_KEY] = confusion.ravel().tolist()
    return agg


class LoggingFedAvg(FedAvg):
    """Plain FedAvg that records every client reply before aggregating."""

    def __init__(self, logger: JsonlLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.site_visualizations: dict[int, list[dict]] = {}

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
        points = []
        for msg in replies:
            if msg.has_error():
                continue
            metrics = next(iter(msg.content.metric_records.values()))
            embeddings, y_true, y_pred = msg.content[
                "visualization"
            ].to_numpy_ndarrays()
            points.append(
                {
                    "site": f"client {metrics['partition-id']}",
                    "source": "site",
                    "embeddings": embeddings,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        self.site_visualizations[server_round] = points
        agg = super().aggregate_evaluate(server_round, replies)
        if agg is not None:
            self.logger.write("federated_eval", round=server_round, **dict(agg))
        return agg


def _initial_arrays(cfg: dict) -> ArrayRecord:
    """The starting weights. A warm start loads a model trained elsewhere.

    Every arm of the comparison has to start from the same weights, so a run
    that warm starts the federation must warm start the local and pooled
    baselines too.
    """
    model = build_model(cfg)
    path = str(cfg.get("init-model", "")).strip()
    if path:
        p = Path(os.path.expanduser(path))
        if not p.is_file():
            raise FileNotFoundError(f"No warm start model at {p}.")
        model.load_state_dict(torch.load(p, weights_only=True))
        print(f"Warm start from {p}", flush=True)
    return ArrayRecord(model.state_dict())


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    logger = JsonlLogger(str(cfg["out-dir"]), cfg)
    print(f"Writing metrics to {logger.path}", flush=True)
    logger.write("labels", labels=list(LABELS))

    torch.manual_seed(int(cfg["seed"]))
    min_nodes = int(cfg["min-nodes"])
    embedding_sample = int(cfg["umap-sample-per-site"])
    evaluation_loader, evaluation_labels, evaluation_datasets = load_evaluation(
        cfg["evaluation-file"], cfg
    )
    if evaluation_labels != list(LABELS):
        raise ValueError("The evaluation partition label contract does not match the app.")
    strategy = LoggingFedAvg(
        logger,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
        train_metrics_aggr_fn=_drop_partition_id,
        evaluate_metrics_aggr_fn=_federated_macro_f1,
    )

    def central_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(cfg)
        model.load_state_dict(arrays.to_torch_state_dict())
        details = evaluate_detailed(
            model,
            evaluation_loader,
            len(LABELS),
            max_embedding_cells=embedding_sample,
            seed=int(cfg["seed"]) + server_round,
        )
        n = int(details.confusion.sum())
        accuracy = float(np.trace(details.confusion) / max(n, 1))
        global_f1 = macro_f1(details.tp, details.fp, details.fn)
        logger.write(
            "external_eval",
            round=server_round,
            **{
                "num-examples": n,
                "eval_loss": details.loss,
                "accuracy": accuracy,
                "macro_f1": global_f1,
            },
        )
        logger.write(
            "external_confusion",
            round=server_round,
            scope="global",
            confusion=details.confusion.ravel().tolist(),
        )

        visualizations = list(strategy.site_visualizations.get(server_round, []))
        for group_id, confusion in details.group_confusions.items():
            dataset_name = evaluation_datasets[group_id]
            logger.write(
                "external_confusion",
                round=server_round,
                scope=dataset_name,
                confusion=confusion.ravel().tolist(),
            )
            if details.groups is None:
                continue
            mask = details.groups == group_id
            visualizations.append(
                {
                    "site": dataset_name,
                    "source": "external",
                    "embeddings": details.embeddings[mask],
                    "y_true": details.y_true[mask],
                    "y_pred": details.y_pred[mask],
                }
            )

        embedding_parts = [
            item["embeddings"] for item in visualizations if len(item["embeddings"])
        ]
        embeddings = (
            np.concatenate(embedding_parts, axis=0)
            if embedding_parts
            else np.empty((0, int(cfg["hidden-size"])), dtype=np.float32)
        )
        if len(embeddings) >= 3:
            coordinates = UMAP(
                n_components=2,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric="euclidean",
                random_state=int(cfg["seed"]),
                n_jobs=1,
            ).fit_transform(embeddings)
            points = []
            offset = 0
            for item in visualizations:
                count = len(item["embeddings"])
                for i, (x, y) in enumerate(coordinates[offset : offset + count]):
                    true = int(item["y_true"][i])
                    predicted = int(item["y_pred"][i])
                    points.append(
                        {
                            "x": float(x),
                            "y": float(y),
                            "label": true,
                            "predicted": predicted,
                            "correct": true == predicted,
                            "source": item["source"],
                            "site": item["site"],
                        }
                    )
                offset += count
            logger.write_umap(server_round, points)
        else:
            logger.write("umap_unavailable", round=server_round, points=len(embeddings))

        return MetricRecord(
            {
                "num-examples": n,
                "eval_loss": details.loss,
                "accuracy": accuracy,
                "macro_f1": global_f1,
            }
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=_initial_arrays(cfg),
        train_config=ConfigRecord({"lr": float(cfg["lr"])}),
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=central_eval,
    )

    torch.save(result.arrays.to_torch_state_dict(), logger.dir / "final_model.pt")
    logger.write("run_end")
    print(f"Done. Metrics: {logger.path}  Model: {logger.dir / 'final_model.pt'}", flush=True)
