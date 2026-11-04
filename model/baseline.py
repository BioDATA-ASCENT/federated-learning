"""the main results table: local vs FedAvg vs pooled, per client.
"""

import numpy as np
import pandas as pd

from track_a.task import (DATA_DIR, LogisticRegressionModel, client_files, get_labels,
                          load_data, test, train)

ROUNDS, EPOCHS = 20, 1          # must match the Flower run
TOTAL_EPOCHS = ROUNDS * EPOCHS  # local and pooled get the same budget

labels = get_labels()
clients = [load_data(i) for i in range(len(client_files()))]   # (X_tr, y_tr, X_te, y_te, name)
n_genes = clients[0][0].shape[1]

# FedAvg: the final global model from the Flower run
fed = LogisticRegressionModel(len(labels), n_genes)
w = np.load(DATA_DIR / "final_weights.npz")
fed.set_weights([w["coef"], w["intercept"]])

# Pooled: one model on all clients' training cells together
pooled = LogisticRegressionModel(len(labels), n_genes)
X_all = np.concatenate([c[0] for c in clients])
y_all = np.concatenate([c[1] for c in clients])
train(pooled, X_all, y_all, TOTAL_EPOCHS)

rows = []
for X_tr, y_tr, X_te, y_te, name in clients:
    # Local: this client alone
    local = LogisticRegressionModel(len(labels), n_genes)
    train(local, X_tr, y_tr, TOTAL_EPOCHS)
    rows.append({
        "client": name, "n_train": len(y_tr),
        "local": test(local, X_te, y_te)[0],
        "fedavg": test(fed, X_te, y_te)[0],
        "pooled": test(pooled, X_te, y_te)[0],
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format="%.3f"))
table.to_csv(DATA_DIR / "results_table.csv", index=False)
