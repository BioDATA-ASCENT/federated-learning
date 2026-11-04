# Federated learning for biological data

Train one model across many labs or hospitals without moving the data. Each client fits the model on its own data and sends only the weights to the server. The server averages the weights and sends them back. Hackathon project from BioDATA ASCENT at CSHL, Nov 2026. Team of six.

Start with [quickstart.md](quickstart.md). It sets up the environment and downloads the data. [Progress.md](Progress.md) is the checklist. [DEPLOYMENT.md](DEPLOYMENT.md) covers running the federation across machines: one server, laptops as nodes. [SITE_ONBOARDING.md](SITE_ONBOARDING.md) is the non-technical guide to show a new partner site, such as a hospital or lab, when it joins.

### Federation server

The scripts and docs read the server login from two environment variables:

```bash
export FL_HOST=100.53.208.149   # federation server (AWS)
export FL_USER=user1            # default shared login; use your own if the lead gave you one
```

The password is never stored in the repo. Ask the lead for it, and use it once with `ssh-copy-id` (see [DEPLOYMENT.md](DEPLOYMENT.md)).

## Food for thought

Start with the FedAvg paper and your track's references below. These questions examine the proposed design:

- Does averaging locally trained logistic regressions reproduce pooled training? How do local epochs, initialization, and learning rates affect the comparison?
- Which preprocessing steps use pooled or held-out data? What does that imply for the federation and evaluation?
- How should missing classes enter macro F1? Is averaging client scores equivalent to evaluating pooled predictions?
- If attribution is the global coefficient vector, does computing it at every client add information? What would make an attribution data-dependent?

Choose a track and one question to explore with a small prototype. The three stages below give a common direction. How far you take them is your choice.

## The three stages

The tracks can share one FedAvg loop, one client interface, and one result table.

1. **Federated training.** Client: receive global weights, train locally for E epochs, return weights and sample count. Server: weighted average, repeat for R rounds. Crosses the wire: weights only.
2. **Federated inference.** Client: evaluate the global model on local held-out data, return the metric. Server: report per client and as a weighted mean. Output: the main table (rows = clients, columns = local, FedAvg, pooled) and one plot of per-client gain.
3. **Federated interpretability.** Client: compute feature attributions with the global model on local data, return the vector. Server: average. Output: top features from the federated model versus top features from each local model.

## The contracts

FL works only when every client agrees on four things before the first round. Make each one explicit, because this is where real federations fail.

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
| Human Lung Cell Atlas core (Sikkema et al. 2023) | Track A, the work | Six, one per lab group | 584,944 cells, 27,402 genes, 5.87 GB h5ad | [CELLxGENE](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) |
| cBioPortal studies | Tracks B and C | One per study | Hundreds to thousands of samples per study, fetched by API | [cBioPortal](https://www.cbioportal.org) |

The pancreas data are small and come from nine different protocols. That is the same multi-lab structure as the HLCA, at one thirty-fifth the cell count. Build and test on it first. scFed (Wang et al. 2024) also benchmarked on pancreas data, so its reported numbers are a reference point.

## Tracks

| Track | Task | Detail |
|---|---|---|
| A | Cell type classification on the Human Lung Cell Atlas, one client per contributing lab | [tracks/TrackA_scRNAseq_annotation.md](tracks/TrackA_scRNAseq_annotation.md) |
| B | Cancer tissue of origin from panel mutations, one client per institution | [tracks/TrackB_tissue_of_origin.md](tracks/TrackB_tissue_of_origin.md) |
| C | Survival on immune checkpoint inhibitors, one client per hospital cohort | [tracks/TrackC_ici_survival.md](tracks/TrackC_ici_survival.md) |

Track A is the work. The team chose cell type classification on the HLCA, with nine `ann_level_2` classes and six clients. Tracks B and C stay here as written, unstarted.

## Before the hackathon

The organizers ask for a full prototype by noon, so a pipeline must run end to end before the day starts. [Progress.md](Progress.md) lists the parts and the condition that closes each one.

The repository holds the environment, the download script, these docs, and two Flower apps. [apps/mnist_fedavg](apps/mnist_fedavg) is the toy FedAvg app that proves the deployment path. [apps/hlca_fedavg](apps/hlca_fedavg) is Track A: it runs five rounds over six clients in simulation, and the partitions it reads are cut by hand, outside the app.

Framework: Flower, `flwr[simulation]==1.42.0`, Python 3.11 or 3.12, Message API only (`ServerApp`, `ClientApp`, `Grid`, `@app.train()`). Tutorials that use `NumPyClient`, `server_fn`, or `start_simulation` are the old API. Do not follow them.

## The day

Two times are fixed: a checkpoint at noon and the presentation at 5:00. Use the rest as you see fit.

- Morning: read, discuss the questions above and in your track, and sketch the smallest useful experiment.
- Noon: show what runs and what you learned. Choose what to do next.
- Afternoon: run the experiment and compare it with a baseline.
- 3:00: keep a working run and its saved outputs.
- 5:00: present the result and the next experiment you would try.

A negative result is useful. If only part of the work runs, show that part and say what remains.

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
