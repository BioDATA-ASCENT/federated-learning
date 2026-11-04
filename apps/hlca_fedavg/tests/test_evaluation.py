from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hlca_fedavg.task import MLP, evaluate_detailed, is_umap_round


def test_umap_capture_is_scheduled_every_five_training_rounds():
    assert [rnd for rnd in range(16) if is_umap_round(rnd)] == [5, 10, 15]


def test_evaluate_detailed_counts_every_cell_and_caps_each_group():
    torch.manual_seed(7)
    features = torch.randn(20, 4)
    labels = torch.arange(20) % 3
    groups = torch.tensor([0] * 11 + [1] * 9)
    loader = DataLoader(
        TensorDataset(features, labels, groups),
        batch_size=4,
        shuffle=False,
    )
    model = MLP(n_genes=4, hidden=8, n_classes=3)

    result = evaluate_detailed(
        model, loader, n_classes=3, max_embedding_cells=3, seed=19
    )
    repeat = evaluate_detailed(
        model, loader, n_classes=3, max_embedding_cells=3, seed=19
    )

    assert result.confusion.sum() == len(labels)
    assert sum(matrix.sum() for matrix in result.group_confusions.values()) == len(labels)
    assert result.embeddings.shape == (6, model.fc1.out_features)
    assert np.bincount(result.groups, minlength=2).tolist() == [3, 3]
    rng = np.random.default_rng(19)
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(groups.numpy() == group), 3, replace=False)
            for group in np.unique(groups.numpy())
        ]
    )
    selected.sort()
    with torch.no_grad():
        expected_embeddings = model.forward_with_embedding(features)[1].numpy()[selected]
    np.testing.assert_allclose(result.embeddings, expected_embeddings)
    np.testing.assert_array_equal(result.y_true, labels.numpy()[selected])
    np.testing.assert_array_equal(result.embeddings, repeat.embeddings)
    np.testing.assert_array_equal(result.y_true, repeat.y_true)
    np.testing.assert_array_equal(result.y_pred, repeat.y_pred)


def test_evaluate_detailed_includes_every_cell_for_umap():
    torch.manual_seed(11)
    features = torch.randn(20, 4)
    labels = torch.arange(20) % 3
    groups = torch.tensor([0] * 11 + [1] * 9)
    loader = DataLoader(
        TensorDataset(features, labels, groups),
        batch_size=4,
        shuffle=False,
    )
    model = MLP(n_genes=4, hidden=8, n_classes=3)

    result = evaluate_detailed(
        model, loader, n_classes=3, max_embedding_cells=None, seed=19
    )

    with torch.no_grad():
        expected_logits, expected_embeddings = model.forward_with_embedding(features)
    assert result.embeddings.shape == (len(labels), model.fc1.out_features)
    np.testing.assert_allclose(result.embeddings, expected_embeddings.numpy())
    np.testing.assert_array_equal(result.y_true, labels.numpy())
    np.testing.assert_array_equal(result.y_pred, expected_logits.argmax(1).numpy())
    np.testing.assert_array_equal(result.groups, groups.numpy())
    assert sum(matrix.sum() for matrix in result.group_confusions.values()) == len(labels)


def test_evaluate_detailed_caps_ungrouped_sample():
    features = torch.randn(12, 4)
    labels = torch.arange(12) % 2
    loader = DataLoader(TensorDataset(features, labels), batch_size=5, shuffle=False)
    model = MLP(n_genes=4, hidden=8, n_classes=2)

    result = evaluate_detailed(
        model, loader, n_classes=2, max_embedding_cells=5, seed=3
    )

    assert result.confusion.sum() == len(labels)
    assert result.embeddings.shape == (5, model.fc1.out_features)
    assert result.groups is None
    selected = np.random.default_rng(3).choice(len(labels), 5, replace=False)
    selected.sort()
    with torch.no_grad():
        expected_embeddings = model.forward_with_embedding(features)[1].numpy()[selected]
    np.testing.assert_allclose(result.embeddings, expected_embeddings)
    np.testing.assert_array_equal(result.y_true, labels.numpy()[selected])
