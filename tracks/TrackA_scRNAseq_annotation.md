# Track A: cell type classification across labs

Back to [README](../README.md). Data download: [quickstart.md](../quickstart.md).

## Food for thought

Read scFed for the method and HLCA for the data (references below).

- Could cells from one donor appear in both training and testing? What changes if donors are held out instead?
- HVGs are selected before splitting in the proposed recipe. How could test cells influence feature selection, and how would training-only selection change the experiment?
- If clients min-max scale genes separately, does the same coefficient represent the same expression change? What does that mean for averaging weights and comparing gene rankings?
- What evidence turns a predictive coefficient into a marker gene? Are scFed's held-out-dataset scores comparable to our within-client evaluation?

Choose one question, start with a small pancreas prototype, and expand toward HLCA or interpretation as useful.

| Item | Value |
|---|---|
| Task | Cell type classification |
| Data | Human Lung Cell Atlas core (Sikkema et al. 2023), raw counts, 2000 highly variable genes |
| Clients | `dataset` column. 14 contributing labs. Use the 5 or 6 largest for the demo, all 14 for the paper |
| Label | `ann_level_3`. Tens of classes. `ann_finest_level` (61 classes) for the paper |
| Model | Logistic regression |
| Metric | Macro F1 |
| Stage 3 output | Marker genes per cell type, federated versus per client |
| Smoke test data | scIB human pancreas, one client per `tech` value |
| Status | Parallel track choice |

## Data

Two real datasets. The pipeline is written once and runs on both. The track name selects the dataset: `load_client("pancreas", k)` or `load_client("hlca", k)`.

### Pancreas (smoke test)

The scIB human pancreas benchmark: 16,382 cells, 19,093 genes, nine protocols (celseq, celseq2, fluidigmc1, inDrop1 to inDrop4, smarter, smartseq2) in `obs["tech"]`, cell types in `obs["celltype"]`, raw counts in `layers["counts"]`. One client per `tech` value. The file downloads in minutes and the full pipeline runs on a laptop. scFed also benchmarked on pancreas data with one dataset held out per run, so its reported F1 is a reference point for the linear model.

### HLCA core (demo)

584,944 cells, 28,024 genes, 107 donors, 14 datasets, CC BY 4.0. The h5ad on CELLxGENE is 5.9 GB (dataset id `066943a2-fdac-4b29-b348-40cede398e4e`). The lead downloads it once. The team never touches it. Raw counts are in `adata.raw.X`. Metadata description: https://github.com/LungCellAtlas/HLCA (docs/HLCA_metadata_explanation.csv).

### Partitions

`make_partitions.py` runs once per dataset and writes `data/<track>_client_K.h5ad`, a few MB each:

1. Read the source file. For HLCA, keep the 5 or 6 largest `dataset` values. For pancreas, keep all nine `tech` values.
2. Compute HVGs on the union with `batch_key` set to the client column, so the gene set is shared. This is the one preprocessing step that runs on pooled data, and it is declared as such.
3. Subset to the 2000 HVGs in a fixed order.
4. Write one h5ad per client with raw counts and the client and label columns.

## What scFed gives us and what it does not

scFed is a custom socket-based framework, not Flower, so none of its FL code transfers. Code: https://github.com/digi2002/federatedSinglecell. Three things transfer:

1. The recipe. Drop all-zero genes, CPM normalize, log2(x+1), min-max scale per client. Its linear SVM is scikit-learn `SGDClassifier` with hinge loss, aggregated by a sample-weighted average of `coef_` and `intercept_`. Our logistic regression is the same aggregation with log loss, so Stage 1 reproduces scFed's linear model in Flower.
2. The baseline numbers. scFed reports F1 for the linear model and ACTINN on pancreas with one dataset held out per run. We cite them. We do not rerun them.
3. The cross-dataset design. One client per source dataset, shared cell types only. HLCA replaces pancreas, `ann_level_3` replaces the four pancreas types.

## Contracts for this track

| Contract | How Track A meets it |
|---|---|
| Feature | `make_partitions.py` fixes 2000 genes in one order. `load_client` asserts the count and order. |
| Label | `LABELS` is the union of the label column across the chosen clients, fixed in `fedbio/data.py`. A client missing a class still returns a weight matrix of shape (n_labels, 2000). |
| Preprocessing | HVG selection on the union, once, declared. CPM, log, and scaling per client. Normalization statistics differ by client. That is the non-IID feature shift FedAvg must absorb. |
| Model | `get_weights` returns `[coef_, intercept_]`. Shapes (n_labels, 2000) and (n_labels,). Stage 3 attribution is `coef_` per class. |

## Flower setup

`client_app.py` reads `partition-id` from the node config, calls `load_client(track, k)`, and the `@app.train()` handler fits for E epochs from the received weights. `server_app.py` calls `Grid.send_and_receive` per round and averages by sample count. The same app runs in simulation (`flwr run .`) and on six laptops (`flwr run . lab-federation`) with no code change.

## Possible prototype milestones

Choose milestones to suit your question; these are starting points for exploration.

| ID | Task | Done when |
|---|---|---|
| Prototype | HLCA partitions on every laptop | `load_client("hlca", k)` returns HLCA data in under 10 s |
| Explore | Stage 3 | Top 10 genes per cell type, federated and per client, one table |

## References

- Sikkema et al. 2023, [An integrated cell atlas of the lung in health and disease](https://www.nature.com/articles/s41591-023-02327-2), Nature Medicine. The HLCA. Code and metadata: https://github.com/LungCellAtlas/HLCA
- Luecken et al. 2022, [Benchmarking atlas-level data integration in single-cell genomics](https://www.nature.com/articles/s41592-021-01336-8), Nature Methods. The scIB benchmark. Source of the pancreas file.
- Wang et al. 2024, [scFed: federated learning for cell type classification with scRNA-seq](https://academic.oup.com/bib/article/25/1/bbad507/7529147), Briefings in Bioinformatics. Linear SVM, ACTINN, XGBoost, Geneformer under a custom socket FL framework. Code: https://github.com/digi2002/federatedSinglecell
- [FedscGen](https://www.rna-seqblog.com/fedscgen-privacy-preserving-federated-batch-effect-correction-of-single-cell-rna-sequencing-data/) (2024). Federated batch correction.
