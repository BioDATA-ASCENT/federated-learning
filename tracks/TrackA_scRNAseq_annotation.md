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
| Clients | Six, one per lab group. The two largest labs train the initial model and two more are held out for the deployment demo |
| Label | `ann_level_2`, nine classes. `LABELS` in `apps/hlca_fedavg/hlca_fedavg/task.py` is the contract |
| Model | MLP, 2000 inputs to 256 hidden units to nine logits |
| Metric | Macro F1 |
| Stage 3 output | Marker genes per cell type, federated versus per client |
| Smoke test data | scIB human pancreas, one client per `tech` value |
| Status | Active. Code in `apps/hlca_fedavg/` (MLP) and `apps/hlca_logreg/` (logistic regression) |

## Data

The HLCA core holds every client's cells in one file, labeled by `obs["dataset"]`. The partitions are cut from it by hand, outside the app.

### Pancreas (smoke test)

The scIB human pancreas benchmark: 16,382 cells, 19,093 genes, nine protocols (celseq, celseq2, fluidigmc1, inDrop1 to inDrop4, smarter, smartseq2) in `obs["tech"]`, cell types in `obs["celltype"]`, raw counts in `layers["counts"]`. One client per `tech` value. The file downloads in minutes and the full pipeline runs on a laptop. scFed also benchmarked on pancreas data with one dataset held out per run, so its reported F1 is a reference point for the linear model.

### HLCA core (demo)

584,944 cells, 27,402 genes, 107 donors, 14 datasets, CC BY 4.0. The h5ad on CELLxGENE is 5.87 GB (dataset id `066943a2-fdac-4b29-b348-40cede398e4e`). The lead downloads it once. The team never touches it. Raw counts are in `adata.raw.X`. Metadata description: https://github.com/LungCellAtlas/HLCA (docs/HLCA_metadata_explanation.csv).

### Partitions

Somebody cuts the partitions by hand, outside the app. Each node then holds its own copy in `~/fedbio-data/hlca/`. What the app needs:

1. Six client files, each one lab group, named `client_0.h5ad` to `client_5.h5ad`.
2. `genes.txt`, 2000 highly variable genes computed from one partition alone, one symbol per line. No step runs on pooled data.
3. Every client file subset to those genes, in that order, carrying raw counts.
4. `obs["ann_level_2"]` for the label and `obs["donor_id"]` for the split.

The cells of the two largest labs go to the initial model and two more labs are held out for the deployment demo, so the six clients hold the remaining ten `dataset` values, 253,208 cells in nine classes.

## What scFed gives us and what it does not

scFed is a custom socket-based framework, not Flower, so none of its FL code transfers. Code: https://github.com/digi2002/federatedSinglecell. Three things transfer:

1. The recipe. Drop all-zero genes, CPM normalize, log2(x+1), min-max scale per client. Its linear SVM is scikit-learn `SGDClassifier` with hinge loss, aggregated by a sample-weighted average of `coef_` and `intercept_`. Our logistic regression is the same aggregation with log loss, so Stage 1 reproduces scFed's linear model in Flower.
2. The baseline numbers. scFed reports F1 for the linear model and ACTINN on pancreas with one dataset held out per run. We cite them. We do not rerun them.
3. The cross-dataset design. One client per source lab, shared cell types only.

## Contracts for this track

| Contract | How Track A meets it |
|---|---|
| Feature | `genes.txt` fixes 2000 genes in one order, computed from one partition. `load_client` asserts the count and the order, and raises on a mismatch. |
| Label | `LABELS` holds nine `ann_level_2` classes in a fixed order. Five of them are absent from at least one client, and a client missing a class still returns the full weight shape. |
| Preprocessing | No step runs on pooled data, because the gene list comes from one partition. `normalize_total` to 1e4 and `log1p` run per client, so the normalization statistics differ by client. That is the feature shift FedAvg must absorb. |
| Model | `build_model` is the one place that fixes the shapes, so every node agrees. `ArrayRecord(model.state_dict())` carries the four tensors. The MLP has no per-gene coefficient, so Stage 3 needs gradient attribution. |

## Flower setup

`apps/hlca_fedavg/` holds the app. See its README. The client reads `partition-id` from the node config, loads that client's partition, and fits for E epochs from the received weights in an `@app.train()` handler. The server runs `FedAvg` and averages by sample count. `LoggingFedAvg` records every client reply before aggregating, so the aggregation stays visible. The same app runs in simulation and on six laptops.

## Possible prototype milestones

Choose milestones to suit your question; these are starting points for exploration.

| ID | Task | Done when |
|---|---|---|
| Prototype | HLCA partitions on every laptop | `flwr run apps/hlca_fedavg` completes a round on every laptop |
| Explore | Stage 3 | Top 10 genes per cell type, federated and per client, one table |

## References

- Sikkema et al. 2023, [An integrated cell atlas of the lung in health and disease](https://www.nature.com/articles/s41591-023-02327-2), Nature Medicine. The HLCA. Code and metadata: https://github.com/LungCellAtlas/HLCA
- Luecken et al. 2022, [Benchmarking atlas-level data integration in single-cell genomics](https://www.nature.com/articles/s41592-021-01336-8), Nature Methods. The scIB benchmark. Source of the pancreas file.
- Wang et al. 2024, [scFed: federated learning for cell type classification with scRNA-seq](https://academic.oup.com/bib/article/25/1/bbad507/7529147), Briefings in Bioinformatics. Linear SVM, ACTINN, XGBoost, Geneformer under a custom socket FL framework. Code: https://github.com/digi2002/federatedSinglecell
- [FedscGen](https://www.rna-seqblog.com/fedscgen-privacy-preserving-federated-batch-effect-correction-of-single-cell-rna-sequencing-data/) (2024). Federated batch correction.
