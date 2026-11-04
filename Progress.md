# Progress

Check completed preparation and chosen milestones. Add the date and your initials. Use the README and track docs for experiment ideas.

## Before the hackathon

- [ ] 1. Decide a name for the project. The name sets the repository name, the Python package name (now `fedbio`), and the Flower app name (now `fedbio_app`).
- [x] 2. Environment: `uv sync` works on a clean laptop. `python -c "import flwr"` prints `1.42.0`. ([quickstart.md](quickstart.md)) 2026-11-03 MV
- [x] 3. Pancreas data downloaded on every laptop. 2026-11-03 MV (lead's laptop. 16,382 cells, 19,093 genes, 9 techs.)
- [x] 4. HLCA core downloaded on the lead's laptop. 2026-11-03 MV (5.87 GB, 584,944 cells, 27,402 genes, 14 datasets.)
- [ ] 5. Partitions: one h5ad per `tech` value from the pancreas file.
- [ ] 6. Client loader: returns `(X_train, y_train, X_test, y_test)` for one pancreas client, with a fixed gene order and the full label set.
- [ ] 7. Model: fit, get weights, set weights, evaluate, and attribute, on logistic regression.
- [ ] 8. FedAvg loop: local, FedAvg, and pooled on the pancreas data. Writes the result table.
- [ ] 9. Flower app: a simulation run on the pancreas data matches the FedAvg loop within one point of macro F1.
- [ ] 10. The per-client gain figure renders from the result table.
- [ ] 11. Stage 3 on pancreas: top 10 genes per cell type, federated and per client.
- [ ] 12. One command runs items 5 to 11 end to end in under five minutes.
- [ ] 13. Partitions for the 5 or 6 largest HLCA datasets, one file per client. Files shared with the team.

## Hackathon checkpoints

- [ ] Share ideas from the literature and discuss the chosen track's questions.
- [ ] Choose a question and the smallest useful prototype, individually or in pairs.
- [ ] Noon: show what runs, what was learned, and what remains; choose an afternoon experiment.
- [ ] Run the experiment and prepare a table or plot with its assumptions and limitations.
- [ ] 3:00: preserve the working demo and saved outputs.
- [ ] 4:00: rehearse.
- [ ] 5:00: present the result and next experiment.

## After the hackathon

- [ ] Six-laptop deployment with the Flower Deployment Engine (one SuperLink, one SuperNode per laptop).
- [ ] Differential privacy with `LocalDpMod`.
- [ ] Secure aggregation.
- [ ] Track A on all 14 HLCA datasets with `ann_finest_level`.
- [ ] Track C on AACR Project GENIE BPC (needs Synapse registration and a data use agreement).
- [ ] Resource paper: the three-stage protocol across task types on open data, in simulation and deployment.
