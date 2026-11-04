# hlca_logreg

FedAvg logistic regression for cell type classification on the Human Lung Cell
Atlas. It is the second Track A app beside [hlca_fedavg](../hlca_fedavg): same
partitions, same gene list, same nine `ann_level_2` classes, a linear model in
place of the MLP. The linear model gives per-class gene weights directly, which
is the Stage 3 interpretation: `top_genes_federated.csv` lists the 10 genes
with the largest positive weight per cell type.

## Files

| File | Holds |
|---|---|
| `hlca_logreg/task.py` | `LABELS`, the loader and its contract checks, the model, train, test, `macro_f1`, `top_genes` |
| `hlca_logreg/client_app.py` | `@app.train()` and `@app.evaluate()` |
| `hlca_logreg/server_app.py` | `@app.main()`, `LoggingFedAvg`, the federated macro F1 |
| `baseline.py` | The results table: local vs FedAvg vs pooled, per client |
| `tests/test_contracts.py` | The contracts, and the checks that reject a broken partition |

## What each node needs

The same files as `hlca_fedavg`, in `~/fedbio-data/hlca/`. `scripts/partition_hlca.py`
writes them:

```
~/fedbio-data/hlca/genes.txt        one gene per line, the feature contract
~/fedbio-data/hlca/client_<k>.h5ad  raw counts over those genes, in that order,
                                    plus obs["ann_level_2"] and obs["donor_id"]
```

A node reads `client_<partition-id>.h5ad`, never a glob over the directory, so
`evaluation.h5ad` and `initial_training.h5ad` are never treated as clients.
Without a `partition-id` in the node config, a node takes it from the name of
the one `client_<k>.h5ad` it holds.

Preprocessing is per cell only: log2(CPM + 1), with the library size taken as
the row sum over the contract genes. There is no per-client scaling, because
FedAvg needs every client in one feature space. Validation holds out whole
donors.

`LABELS` is copied from `hlca_fedavg`, because a FAB ships only its own
package. `test_labels_match_the_other_app` fails if the two copies drift.

## Run it

```bash
flwr federation simulation-config @none/default --num-supernodes 6
flwr run apps/hlca_logreg --stream
```

Check that the log reports six nodes per round (`[round r] 6 nodes`) and a
score for every class. Override anything in `[tool.flwr.app.config]` at
submit time:

```bash
flwr run apps/hlca_logreg --stream -c 'num-server-rounds=10 lr=0.005'
```

## Output

Under `~/fedbio-runs/hlca_logreg/<timestamp>/`:

| File | Holds |
|---|---|
| `metrics.jsonl` | One line per client reply per round, plus a `federated_eval` line per round with macro F1, per-class F1 and the summed counts |
| `final_weights.npz` | `coef` (9 × genes) and `intercept` (9) of the last global model |
| `top_genes_federated.csv` | The top 10 genes per cell type |

The clients return per-class true positive, false positive and false negative
counts, not a score. The server sums them and computes macro F1 once, over the
classes that have a true cell.

## The results table

```bash
uv run python apps/hlca_logreg/baseline.py            # newest run under out-dir
uv run python apps/hlca_logreg/baseline.py --weights ~/fedbio-runs/hlca_logreg/<timestamp>/final_weights.npz
```

Run it where every `client_<k>.h5ad` is present. All three arms start from zero
weights. Local and pooled train for rounds × local epochs, read from the run's
`metrics.jsonl`, so they get the same budget as the federation. The last row
sums the counts over clients, which makes it comparable to the run's
`macro_f1`. The table is saved as `results_table.csv` beside the weights.

The MLP app can warm start from `initial.pt`, but this app always starts from
zero. Compare it against a cold-started `hlca_fedavg` run.

## Tests

```bash
FEDBIO_DATA=~/fedbio-data/hlca uv run pytest apps/hlca_logreg/tests -q
```

The data tests use the lowest-numbered `client_<k>.h5ad` under `FEDBIO_DATA`
and corrupt copies of it. They skip when no partition is present. The label,
filename and split tests always run.
