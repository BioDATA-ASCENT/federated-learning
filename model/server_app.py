"""server_app.py — what the server runs. Same shape as the quickstart's
server_app.py: build the starting model, hand it to the FedAvg strategy,
save the final model."""

import numpy as np
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from track_a.task import DATA_DIR, client_files, get_labels, top_genes
import anndata as ad

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 20)
    local_epochs = context.run_config.get("local-epochs", 1)

    # 1. The starting model: all weights zero, with the full shape.
    labels = get_labels()
    genes = list(ad.read_h5ad(client_files()[0], backed="r").var_names)
    initial = ArrayRecord([np.zeros((len(labels), len(genes))), np.zeros(len(labels))])

    # 2. FedAvg: each round, send weights to every client, let each train,
    #    then average the returned weights, weighted by "num-examples".
    n_clients = len(client_files())
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      min_available_nodes=n_clients, min_train_nodes=n_clients,
                      min_evaluate_nodes=n_clients)
    result = strategy.start(
        grid=grid,
        initial_arrays=initial,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": local_epochs}),
    )

    # 3. Save the final global model.
    coef, intercept = result.arrays.to_numpy_ndarrays()
    np.savez(DATA_DIR / "final_weights.npz", coef=coef, intercept=intercept)

    # 4. F1 per round (averaged over clients, weighted by test size).
    rows = [{"round": r, **dict(m)} for r, m in result.evaluate_metrics_clientapp.items()]
    print(pd.DataFrame(rows)[["round", "macro-f1", "accuracy"]].to_string(index=False))

    # 5. Stage 3: top 10 genes per cell type from the global model.
    tops = top_genes(coef, genes, labels, k=10)
    pd.DataFrame(tops).to_csv(DATA_DIR / "top_genes_federated.csv", index=False)
    print("saved final_weights.npz and top_genes_federated.csv in", DATA_DIR)
