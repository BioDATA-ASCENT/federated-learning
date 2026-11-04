# Progress

One line per item. Check the box when the "done when" condition in the README holds. Add the date and your initials.

## Before the hackathon

- [ ] 1. Decide a name for the project. The name sets the repository name, the Python package name (now `fedbio`), and the Flower app name (now `fedbio_app`).
- [x] 2. Environment: `uv sync` works on a clean laptop. `python -c "import flwr"` prints `1.42.0`. ([quickstart.md](quickstart.md)) 2026-11-03 MV
- [ ] 3. Pancreas data downloaded on every laptop.
- [ ] 4. HLCA core downloaded on the lead's laptop.
- [ ] 5. `make_partitions.py` writes one h5ad per `tech` value from the pancreas file.
- [ ] 6. `fedbio/data.py`: `load_client("pancreas", k)` returns `(X_train, y_train, X_test, y_test)` with a fixed gene order and the full label set.
- [ ] 7. `fedbio/model.py`: `fit`, `get_weights`, `set_weights`, `evaluate`, `attribute` on logistic regression.
- [ ] 8. `fedbio/fedavg.py`: local, FedAvg, and pooled on the pancreas data. Writes `results/pancreas_table.csv`.
- [ ] 9. `fedbio_app/`: `flwr run .` on the pancreas data matches `fedavg.py` within one point of macro F1.
- [ ] 10. `plot_gain.py` renders the per-client gain figure from `results/pancreas_table.csv`.
- [ ] 11. Stage 3 on pancreas: top 10 genes per cell type, federated and per client.
- [ ] 12. `make test` runs items 5 to 11 end to end in under five minutes.
- [ ] 13. `make_partitions.py` writes `data/hlca_client_K.h5ad` for the 5 or 6 largest HLCA datasets. Files shared with the team.

## Hackathon morning

- [ ] M1. `load_client("hlca", k)` returns HLCA data on every laptop in under 10 s.
- [ ] M2. `load_client("tissue", k)` and `load_client("ici", k)` return data for 3 clients each.
- [ ] M3. Own rewrite of `fedavg.py` gives the same numbers as the prototype on the pancreas data.
- [ ] M4. `flwr run .` on HLCA matches M3 within one point.
- [ ] M5. Gain figure renders from `results/A_table.csv`.
- [ ] 12:00 go/no-go: table, plot, and top genes on HLCA.

## Hackathon afternoon

- [ ] P1. Track A Stage 3: top 10 genes per cell type, federated and per client, one table.
- [ ] P2. Track B: `results/B_table.csv` with local, FedAvg, pooled columns.
- [ ] P3. Track C: `expand_survival` passes a unit test against lifelines on one client. `results/C_table.csv` with C-index per client.
- [ ] P4. Flower prints per-client metrics and averaged attribution.
- [ ] P5. A reader can run the demo from the README. Slides done.
- [ ] 3:00 code freeze.
- [ ] Six-laptop run of Track A (simulation is the backup).
- [ ] 5:00 presentation.

## After the hackathon

- [ ] Six-laptop deployment with the Flower Deployment Engine (one SuperLink, one SuperNode per laptop).
- [ ] Differential privacy with `LocalDpMod`.
- [ ] Secure aggregation.
- [ ] Track A on all 14 HLCA datasets with `ann_finest_level`.
- [ ] Track C on AACR Project GENIE BPC (needs Synapse registration and a data use agreement).
- [ ] Resource paper: the three-stage protocol across task types on open data, in simulation and deployment.
