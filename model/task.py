#data loading and the model

import os
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Settings. Change these if the partition files use other names.
DATA_DIR = Path(os.environ.get("TRACKA_DATA", "data/pancreas"))
LABEL = os.environ.get("TRACKA_LABEL", "celltype")


# ---------------------------------------------------------------------------
# Step 1. Which files and which labels
# ---------------------------------------------------------------------------
def client_files():
    return sorted(DATA_DIR.glob("*.h5ad"))


def get_labels():
    f = DATA_DIR / "labels.txt"
    if f.exists():
        return [l for l in f.read_text().splitlines() if l]
    labels = set()
    for p in client_files():
        labels |= set(ad.read_h5ad(p, backed="r").obs[LABEL].astype(str))
    return sorted(labels)


# ---------------------------------------------------------------------------
# Step 2. Load and preprocess ONE client's data
# ---------------------------------------------------------------------------
def load_data(partition_id):
    labels = get_labels()
    path = client_files()[partition_id]
    adata = ad.read_h5ad(path)

    # Counts -> log2(CPM + 1)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float64)
    lib_size = adata.obs["n_counts"].values if "n_counts" in adata.obs else X.sum(axis=1)
    lib_size = np.where(lib_size == 0, 1, lib_size)
    X = np.log2(X / lib_size[:, None] * 1e6 + 1)

    # Cell-type names -> numbers, using the SHARED label list
    y = np.array([labels.index(c) for c in adata.obs[LABEL].astype(str)])

    # 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2d. Min-max scale each gene to [0, 1], using TRAIN cells only
    lo = X_train.min(axis=0)
    span = X_train.max(axis=0) - lo
    span[span == 0] = 1
    X_train = (X_train - lo) / span
    X_test = np.clip((X_test - lo) / span, 0, 1)

    return X_train, y_train, X_test, y_test, path.stem


# ---------------------------------------------------------------------------
# Step 3. The model
# ---------------------------------------------------------------------------
class LogisticRegressionModel:
    """
    The whole model is two arrays:
        coef       (n_labels x n_genes)  one weight per gene, per cell type
        intercept  (n_labels)            one bias per cell type
    Prediction: score_k = coef[k] . x + intercept[k]; predicted type = largest score.
    """

    def __init__(self, n_labels, n_genes, lr=0.01):
        self.n_labels = n_labels
        self.n_genes = n_genes
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4,
                                 learning_rate="constant", eta0=lr, random_state=42)

        self.clf.partial_fit(np.zeros((1, n_genes)), [0], classes=np.arange(n_labels))
        self.set_weights([np.zeros((n_labels, n_genes)), np.zeros(n_labels)])

    def get_weights(self):
        return [self.clf.coef_.copy(), self.clf.intercept_.copy()]

    def set_weights(self, weights):
        self.clf.coef_ = np.array(weights[0], dtype=np.float64)
        self.clf.intercept_ = np.array(weights[1], dtype=np.float64)

    def fit_one_epoch(self, X, y):
        self.clf.partial_fit(X, y, classes=np.arange(self.n_labels))

    def predict(self, X):
        return self.clf.predict(X)


# ---------------------------------------------------------------------------
# Step 4. Train and test (separate functions, like in the quickstart)
# ---------------------------------------------------------------------------
def train(model, X, y, epochs):
    for _ in range(epochs):
        model.fit_one_epoch(X, y)


def test(model, X, y):
    """Macro F1 over the cell types present in this client's test set, and accuracy."""
    pred = model.predict(X)
    f1 = f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)
    acc = float((pred == y).mean())
    return float(f1), acc


# ---------------------------------------------------------------------------
# Step 5. Interpretation: top genes per cell type
# ---------------------------------------------------------------------------
def top_genes(coef, genes, labels, k=10):
    """For each cell type, the k genes with the largest positive weight."""
    return {lab: [genes[i] for i in np.argsort(-coef[c])[:k]] for c, lab in enumerate(labels)}
