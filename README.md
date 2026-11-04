# Federated learning for biological data

Train one model across many labs or hospitals without moving the data. Each client fits the model on its own data and sends only the weights to the server. The server averages the weights and sends them back. Hackathon project from BioDATA ASCENT at CSHL, Nov 2026. Team of six.

Start with [quickstart.md](quickstart.md). It sets up the environment and downloads the data. [Progress.md](Progress.md) is the checklist.

## Food for thought

Start with the FedAvg paper and your track's method and dataset references below. Use these questions to examine the proposed design before prototyping:

- Does averaging locally trained logistic regressions reproduce pooled training? How do local epochs, initialization, and learning rates affect the comparison?
- Which preprocessing steps use pooled or held-out data? What does that imply for the federation and evaluation?
- How should missing classes enter macro F1? Is averaging client scores equivalent to evaluating pooled predictions?
- If attribution is the global coefficient vector, does computing it at every client add information? What would make an attribution data-dependent?

Choose a track and one question to explore with a small prototype. The three stages below give a common direction; participants choose how far to take them for the demo.

## The three stages

All tracks use one FedAvg loop, one client interface, and one result table.

1. **Federated training.** Client: receive global weights, train locally for E epochs, return weights and sample count. Server: weighted average, repeat for R rounds. Crosses the wire: weights only.
2. **Federated inference.** Client: evaluate the global model on local held-out data, return the metric. Server: report per client and as a weighted mean. Output: the main table (rows = clients, columns = local, FedAvg, pooled) and one plot of per-client gain.
3. **Federated interpretability.** Client: compute feature attributions with the global model on local data, return the vector. Server: average. Output: top features from the federated model versus top features from each local model.

## The contracts

FL works only when every client agrees on four things before the first round. The morning session makes each one explicit, because this is where real federations fail.

| Contract | What every client must agree on | What breaks if violated |
|---|---|---|
| Feature | Same features, same order, same units. Fix the feature list once, outside the clients | Weight matrices do not align. Averaging is silent nonsense. |
| Label | Same class set, same encoding, fixed server-side. A client with a missing class still returns the full weight shape | Client with 8 classes and client with 10 classes cannot average |
| Preprocessing | Which steps run per client (normalize, log, scale) and which run once on the union (HVG selection) | Per-client HVG gives different gene sets. Union scaling leaks data across clients |
| Model | Same architecture, same parameter shapes, same serialization order. Serialize the logistic regression as `[coef_, intercept_]` | Server cannot sum arrays of different shapes |

Two more contracts matter in deployment and are noted, not coded, in the hackathon: the metric contract (every client computes the same metric the same way, on a held-out split the server never sees) and the schema contract (the message format between client and server, which Flower fixes with `ArrayRecord` and `MetricRecord`).

The pancreas data make these contracts concrete. Each track then explores what changes for its own data.

## Data

All data in this project are real and public. The repository holds no synthetic data. Download steps are in [quickstart.md](quickstart.md).

| Dataset | Used for | Clients | Size | Source |
|---|---|---|---|---|
| Human pancreas, 9 technologies (scIB benchmark) | Smoke test and the first end-to-end run | One per `tech` value | 16,382 cells, 19,093 genes | [figshare](https://ndownloader.figshare.com/files/24539828) |
| Human Lung Cell Atlas core (Sikkema et al. 2023) | Track A demo | One per `dataset` value, 14 labs | 584,944 cells, 27,402 genes, 5.87 GB h5ad | [CELLxGENE](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) |
| cBioPortal studies | Tracks B and C | One per study | Hundreds to thousands of samples per study, fetched by API | [cBioPortal](https://www.cbioportal.org) |

The pancreas data are small and come from nine different protocols. That is the same multi-lab structure as the HLCA, at one thirty-fifth the cell count. Build and test on it first. scFed (Wang et al. 2024) also benchmarked on pancreas data, so its reported numbers are a reference point.

## Tracks

| Track | Task | Detail |
|---|---|---|
| A | Cell type classification on the Human Lung Cell Atlas, one client per contributing lab | [tracks/TrackA_scRNAseq_annotation.md](tracks/TrackA_scRNAseq_annotation.md) |
| B | Cancer tissue of origin from panel mutations, one client per institution | [tracks/TrackB_tissue_of_origin.md](tracks/TrackB_tissue_of_origin.md) |
| C | Survival on immune checkpoint inhibitors, one client per hospital cohort | [tracks/TrackC_ici_survival.md](tracks/TrackC_ici_survival.md) |

The tracks are parallel choices. Work individually or in pairs on the problem that interests you; share infrastructure where useful.

## Before the hackathon

The organizers ask for a full prototype by noon. So a pipeline must run end to end on the pancreas data before the hackathon starts. It provides a shared walkthrough and a fallback while participants explore their chosen tracks. [Progress.md](Progress.md) lists the parts to build and the condition that closes each one.

The repository now holds the environment, the download script, and these docs. No pipeline code exists yet.

Framework: Flower, `flwr[simulation]==1.42.0`, Python 3.11 or 3.12, Message API only (`ServerApp`, `ClientApp`, `Grid`, `@app.train()`). Tutorials that use `NumPyClient`, `server_fn`, or `start_simulation` are the old API. Do not follow them.

## Schedule

These are shared checkpoints. Participants choose their experiments and can adjust scope as they learn.

| Time | Block | What happens |
|---|---|---|
| 9:00 | Literature exchange | Share ideas and assumptions from the general and track-specific reading. |
| 9:20 | Brainstorm | Discuss the questions above and in the chosen track. Sketch the smallest useful experiment. |
| 10:30 | Prototype | Walk through the pancreas example or start a track-specific prototype. |
| 12:00 | Checkpoint | Show what runs, what you learned, and what is unresolved. Use the prepared prototype if needed. Choose an afternoon experiment. |
| 1:00 | Explore | Run the experiment and compare with a baseline. Share unexpected results. |
| 3:00 | Demo freeze | Preserve a working run and saved outputs. |
| 4:00 | Rehearse | Explain the question, setup, result, and limitations. |
| 5:00 | Present | Show the demo and the next experiment you would try. |

Possible experiments: write a loader for a chosen track, write a small FedAvg loop, vary local epochs or client balance, compare local/FedAvg/pooled performance, inspect feature weights, or move a working experiment into Flower. Deployment and all three stages remain options; each participant chooses a manageable contribution.

Commit working changes regularly. Ask after 20 minutes stuck. Aim for a runnable comparison and one table or plot per active track. A negative result is useful; if only part of the pipeline works, demonstrate it and explain what remains. Simulation and saved outputs are the demo fallback.

## After the hackathon

Six-laptop deployment with Flower's Deployment Engine (one SuperLink, one SuperNode per laptop). Differential privacy with `LocalDpMod`. Secure aggregation. Nearest resource paper: the three-stage protocol across task types on open data, in simulation and deployment. FLamby (2022) has one survival task and no omics. scFed covers one task. No benchmark we found has a federated inference and interpretation stage.

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
