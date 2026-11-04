from __future__ import annotations

import numpy as np
from flwr.app import MetricRecord, RecordDict

from hlca_fedavg.server_app import _federated_macro_f1
from hlca_fedavg.task import LABELS


def _client_record(partition_id: int, confusion: np.ndarray) -> RecordDict:
    diagonal = np.diag(confusion)
    tp = diagonal
    fp = confusion.sum(axis=0) - diagonal
    fn = confusion.sum(axis=1) - diagonal
    return RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "num-examples": int(confusion.sum()),
                    "partition-id": partition_id,
                    "eval_loss": 0.5,
                    "local_macro_f1": 0.5,
                    "tp": tp.tolist(),
                    "fp": fp.tolist(),
                    "fn": fn.tolist(),
                    "confusion": confusion.ravel().tolist(),
                }
            )
        }
    )


def test_federated_metrics_sum_site_confusion_counts():
    first = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    second = np.zeros_like(first)
    first[0, 0] = 2
    second[0, 1] = 1

    result = _federated_macro_f1(
        [_client_record(0, first), _client_record(1, second)],
        weighting_key="num-examples",
    )

    np.testing.assert_array_equal(
        np.asarray(result["confusion"]).reshape(len(LABELS), len(LABELS)),
        first + second,
    )
    assert result["macro_f1"] == 0.8
