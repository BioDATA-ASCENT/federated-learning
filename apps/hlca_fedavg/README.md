# hlca_fedavg

FedAvg for cell type classification on the Human Lung Cell Atlas. One client per
lab group, nine classes from `ann_level_2`, an MLP over the contract genes.

## Files

| File | Holds |
|---|---|
| `hlca_fedavg/task.py` | `LABELS`, the model, the loader and its contract checks, train, evaluate |
| `hlca_fedavg/client_app.py` | `@app.train()` and `@app.evaluate()` |
| `hlca_fedavg/server_app.py` | `@app.main()`, `LoggingFedAvg`, the federated macro F1 |
| `tests/test_contracts.py` | The contract checks must reject a broken partition |

## What each node needs

Put the partitions and the gene list in `~/fedbio-data/hlca/`:

```
~/fedbio-data/hlca/genes.txt        one gene symbol per line, the feature contract
~/fedbio-data/hlca/client_0.h5ad    raw counts over those genes, in that order
~/fedbio-data/hlca/client_1.h5ad    plus obs["ann_level_2"] and obs["donor_id"]
~/fedbio-data/hlca/evaluation.h5ad  separate held-out evaluation data, with obs["dataset"]
...
```

Each partition carries raw counts. The client normalizes to 1e4 and takes
log1p, which is the per-client half of the preprocessing contract. The loader
asserts the gene order, the label set, that `X` holds counts, and that the two
splits are disjoint. It raises on the first violation and names the file. A
reordered gene list is the reason those checks exist: the matrix widths still
agree, so training and averaging run and produce a meaningless average.

`LABELS` in `task.py` is the label contract. Its length is the output width, so
a client missing a class still returns a weight matrix of the full width.

## Run it

```bash
flwr federation simulation-config @none/default --num-supernodes 6
flwr run apps/hlca_fedavg --stream
```

Override anything in `[tool.flwr.app.config]` at submit time:

```bash
flwr run apps/hlca_fedavg --stream -c 'num-server-rounds=10 local-epochs=2'
```

Keep `local-epochs` at one or two. Hidden units have no fixed order, so
averaging degrades as the clients train further apart.

A warm start loads weights trained elsewhere, for example on the two largest
labs:

```bash
flwr run apps/hlca_fedavg --stream -c 'init-model="~/fedbio-data/initial.pt"'
```

Local, FedAvg and pooled are comparable only when every arm starts from the
same weights. Warm start all three, or none.

## Output

`~/fedbio-runs/hlca_fedavg/<timestamp>/metrics.jsonl` holds one JSON line per
client reply per round, plus the federated line per round. That is the result
table. `final_model.pt` holds the last global weights.

The clients return per-class true positive, false positive and false negative
counts, not a score. A weighted mean of per-client macro F1 is not the macro F1
of the pooled predictions, so the server sums the counts and computes macro F1
once.

The server evaluates each global model on `evaluation.h5ad` every round, as well
as collecting predictions over each client's held-out donors. Both sources are
available in the dashboard's round selector. Confusion matrices are computed
from every test cell; UMAP uses up to 1,500 sampled cells per client and per
external dataset every fifth round. Clients send the sampled 256-dimensional
hidden-layer representations, true labels and predictions to the server only
on those rounds, so it can fit one shared UMAP. The dashboard displays the most
recent fifth-round UMAP at or before the selected round. These derived,
cell-level values (not expression profiles or cell IDs) leave each client. The
UMAP coordinates are saved under `<run>/umap/`; confusion counts and test
metrics are in `metrics.jsonl`.

The central evaluation file must be present on the server at the configured
`evaluation-file` path. Override the default path with Flower's `-c` option
when submitting the run.

## Tests

```bash
FEDBIO_DATA=~/fedbio-data/hlca uv run pytest apps/hlca_fedavg/tests -q
```

They skip when no partition is present. They use a real partition and corrupt
copies of it, because the repository holds no synthetic data.
