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
~/fedbio-data/hlca/client_1.h5ad    raw counts over those genes, in that order
~/fedbio-data/hlca/client_2.h5ad    plus obs["ann_level_2"] and obs["donor_id"]
...                                 up to client_6.h5ad

The files are numbered from 1, as the institutions are. Flower numbers its
partitions from 0, so partition 0 reads `client_1.h5ad`. The log lines name the
file, not the partition id.
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

## Tests

```bash
FEDBIO_DATA=~/fedbio-data/hlca uv run pytest apps/hlca_fedavg/tests -q
```

They skip when no partition is present. They use a real partition and corrupt
copies of it, because the repository holds no synthetic data.
