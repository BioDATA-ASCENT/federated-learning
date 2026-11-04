# Federated learning for biological data

Train one model across many labs or hospitals without moving the data. Each client fits the model on its own data and sends only the weights to the server. The server averages the weights and sends them back. Hackathon project from BioDATA ASCENT at CSHL, Nov 2026. Team of six.

Start with [quickstart.md](quickstart.md). It sets up the environment and downloads the data. [Progress.md](Progress.md) is the checklist.

## The three stages

All tracks share one FedAvg loop, one client interface, and one result table.

1. **Federated training.** Client: receive global weights, train locally for E epochs, return weights and sample count. Server: weighted average, repeat for R rounds. Crosses the wire: weights only.
2. **Federated inference.** Client: evaluate the global model on local held-out data, return the metric. Server: report per client and as a weighted mean. Output: the main table (rows = clients, columns = local, FedAvg, pooled) and one plot of per-client gain.
3. **Federated interpretability.** Client: compute feature attributions with the global model on local data, return the vector. Server: average. Output: top features from the federated model versus top features from each local model.

## The contracts

FL works only when every client agrees on four things before the first round. The morning session makes each one explicit, because this is where real federations fail.

| Contract | What every client must agree on | Where it lives in the code | What breaks if violated |
|---|---|---|---|
| Feature | Same features, same order, same units | `make_partitions.py` fixes the gene list and order. `load_client` asserts `X.shape[1]` | Weight matrices do not align. Averaging is silent nonsense. |
| Label | Same class set, same encoding, fixed server-side | `fedbio/data.py` holds `LABELS`. A client with a missing class still returns the full weight shape | Client with 8 classes and client with 10 classes cannot average |
| Preprocessing | Which steps run per client (normalize, log, scale) and which run once on the union (HVG selection) | Per-client steps in `load_client`. Union steps in `make_partitions.py` | Per-client HVG gives different gene sets. Union scaling leaks data across clients |
| Model | Same architecture, same parameter shapes, same serialization order | `fedbio/model.py`: `get_weights` returns `[coef_, intercept_]` in that order | Server cannot sum arrays of different shapes |

Two more contracts matter in deployment and are noted, not coded, in the hackathon: the metric contract (every client computes the same metric the same way, on a held-out split the server never sees) and the schema contract (the message format between client and server, which Flower fixes with `ArrayRecord` and `MetricRecord`).

Stage 1 of the pipeline is the contract made visible. On the pancreas data, `load_client("pancreas", k)` returns arrays that satisfy all four. The morning task is to make the HLCA data satisfy them.

## Data

All data in this project are real and public. The repository holds no synthetic data. Download steps are in [quickstart.md](quickstart.md).

| Dataset | Used for | Clients | Size | Source |
|---|---|---|---|---|
| Human pancreas, 9 technologies (scIB benchmark) | Smoke test, unit tests, the first end-to-end run | One per `tech` value | 16,382 cells, 19,093 genes | [figshare](https://figshare.com/ndownloader/files/24539828) |
| Human Lung Cell Atlas core (Sikkema et al. 2023) | Track A demo | One per `dataset` value, 14 labs | 584,944 cells, 28,024 genes, 5.9 GB h5ad | [CELLxGENE](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) |
| cBioPortal studies | Tracks B and C | One per study | Hundreds to thousands of samples per study, fetched by API | [cBioPortal](https://www.cbioportal.org) |

The pancreas data are small and come from nine different protocols. That is the same multi-lab structure as the HLCA, at one thirty-fifth the cell count. The pipeline runs on it first. Every test runs on it. scFed (Wang et al. 2024) also benchmarked on pancreas data, so its reported numbers are a reference point.

## Tracks

| Track | Task | Detail |
|---|---|---|
| A | Cell type classification on the Human Lung Cell Atlas, one client per contributing lab | [tracks/TrackA_scRNAseq_annotation.md](tracks/TrackA_scRNAseq_annotation.md) |
| B | Cancer tissue of origin from panel mutations, one client per institution | [tracks/TrackB_tissue_of_origin.md](tracks/TrackB_tissue_of_origin.md) |
| C | Survival on immune checkpoint inhibitors, one client per hospital cohort | [tracks/TrackC_ici_survival.md](tracks/TrackC_ici_survival.md) |

Track A runs in the hackathon. Tracks B and C run in parallel if time allows.

## Before the hackathon

The organizers ask for a full prototype by noon. So the whole pipeline runs end to end on the pancreas data before the hackathon starts. The morning swaps the pancreas data for the HLCA data. Nothing is built from zero on the day.

- `fedbio/data.py`: `load_client(track, k) -> (X_train, y_train, X_test, y_test)`. Tracks: `pancreas`, `hlca`, `tissue`, `ici`.
- `fedbio/model.py`: `fit`, `get_weights`, `set_weights`, `evaluate`, `attribute`. Logistic regression.
- `fedbio/fedavg.py`: the hand-written loop, about 40 lines of NumPy. Runs local, FedAvg, and pooled. Writes `results/<track>_table.csv` and the top-feature table.
- `fedbio_app/`: Flower app from `flwr new`. `server_app.py` uses `Grid.send_and_receive` so the aggregation stays visible. `client_app.py` reads `partition-id` and calls `load_client`.
- `plot_gain.py`: the per-client gain figure.
- `make_partitions.py`: writes `data/<track>_client_K.h5ad` from the pancreas file or from the HLCA core file.
- `make test`: runs the full pipeline on the pancreas data. Target: under five minutes on a laptop.

Framework: Flower, `flwr[simulation]==1.42.0`, Python 3.11 or 3.12, Message API only (`ServerApp`, `ClientApp`, `Grid`, `@app.train()`). Tutorials that use `NumPyClient`, `server_fn`, or `start_simulation` are the old API. Do not follow them.

## Schedule

| Time | Block | What happens |
|---|---|---|
| 9:00 | Kickoff | The three stages and the four contracts on a whiteboard. Then `fedavg.py` on the screen, line by line. |
| 9:20 | Run | Everyone runs `make test`. Each person finds where each contract is enforced in the code. |
| 10:30 | Swap pancreas for HLCA | Five parallel streams, table below. |
| 12:00 | Full prototype | Table, plot, and top genes on HLCA. Go/no-go. |
| 1:00 | Refine | Five parallel streams, table below. |
| 3:00 | Code freeze | |
| 4:00 | Rehearse | |
| 5:00 | Present | |

Rules: commit to main hourly. Ask after 20 minutes stuck. The minimum result is one table.

### Morning, 10:30 to 12:00

| ID | Task | Swaps in | Contract it exercises | Done when |
|---|---|---|---|---|
| M1 | HLCA partitions | HLCA client files | feature, preprocessing | `load_client("hlca", k)` returns HLCA data on every laptop |
| M2 | cBioPortal loader (Tracks B and C) | Track B and C data | feature, label | `load_client("tissue", k)` and `load_client("ici", k)` return data for 3 clients each |
| M3 | Hand-written FedAvg | own rewrite of `fedavg.py` | model | Same numbers as the prototype on the pancreas data |
| M4 | Flower app | HLCA clients in `client_app.py` | model, schema | `flwr run .` matches M3 within 1 point |
| M5 | Evaluation and plots | HLCA table | metric | Gain figure renders from `results/A_table.csv` |

Lead: deployment setup (Tailscale, SuperLink) on the side.

At 12:00: if M3 is not done, the owner uses the prototype loop and keeps the rewrite as a learning artifact.

### Afternoon, 1:00 to 3:00

| ID | Task | From | Done when |
|---|---|---|---|
| P1 | Track A Stage 3 | M1 | Top 10 genes per cell type, federated and per client |
| P2 | Track B all stages | M2 | `results/B_table.csv` with three columns |
| P3 | Track C all stages | M2 and M3 | `expand_survival(df, n_bins)` and `results/C_table.csv` with C-index per client |
| P4 | Flower Stages 2 and 3 | M4 | Flower prints per-client metrics and averaged attribution |
| P5 | README and slides | M5 | A reader can run the demo from the README |

P2 and P3 share the M2 loader. If M2 is late at noon, the P2 and P3 owners join M2 until the loader returns data.

Lead: six-laptop run of Track A at 3:00 if the table exists. Simulation is the backup.

## After the hackathon

Six-laptop deployment with Flower's Deployment Engine (one SuperLink, one SuperNode per laptop, same app code). Differential privacy with `LocalDpMod`. Secure aggregation. Nearest resource paper: the three-stage protocol across task types on open data, in simulation and deployment. FLamby (2022) has one survival task and no omics. scFed covers one task. No benchmark we found has a federated inference and interpretation stage.

## References

### General

- McMahan et al. 2017, [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). The original FedAvg paper. Algorithm 1 is the loop you will write.
- Li et al. 2020, [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873). Short survey, readable in one hour. Covers non-IID data and FedProx.
- Kairouz et al. 2021, [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf). The reference survey, 100+ pages. Use as a lookup, not a read.
- [A review of federated learning: architectures, challenges, and targeted solutions](https://link.springer.com/article/10.1007/s10586-026-06286-4), Cluster Computing 2026.
- Flower docs: [Run Flower with the Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html).

### Biology-specific

- Bakas, Li, Shah, Roth 2026, [Federated Learning in Healthcare: From Research to Real-World Deployment](https://doi.org/10.1146/annurev-bioeng-080125-041414), Annual Review of Biomedical Engineering. The authors ran the FeTS study and built NVIDIA FLARE. Covers non-IID handling, differential privacy, secure aggregation, confidential computing, and deployment.
- Chavero-Diez et al. 2026, [Federated learning frameworks: quality and interoperability for biomedical research](https://doi.org/10.1093/nargab/lqag010), NAR Genomics and Bioinformatics. Systematic comparison of 17 FL frameworks against FAIR software criteria.
- Malpetti et al. 2025, [Technical and legal aspects of federated learning in bioinformatics](https://doi.org/10.3389/fdgth.2025.1644291), Frontiers in Digital Health. Sections on GWAS, proteomics, single-cell and multi-omics.
- [Federated Learning: Breaking Down Barriers in Global Genomic Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11728131/) (2025). Genomics-focused review.

Track-specific references are in each track file.
