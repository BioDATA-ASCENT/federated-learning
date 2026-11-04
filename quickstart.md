# Quickstart

Three steps: install the environment, download the data, run the smoke test. All data are real and public. Nothing here is synthetic.

## 1. Environment

Python 3.11 or 3.12. The environment manager is [uv](https://docs.astral.sh/uv/).

Install uv if `uv --version` fails:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and create the environment:

```bash
git clone https://github.com/BioDATA-ASCENT/federated-learning.git
cd federated-learning
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

`uv venv --python 3.12` downloads Python 3.12 if your machine does not have it. Use `--python 3.11` if you prefer 3.11. `uv sync` installs the packages listed in `pyproject.toml` into `.venv`.

Check the install:

```bash
python -c "import flwr, scanpy, sklearn; print(flwr.__version__)"
```

Expected output: `1.42.0`.

To add a package, run `uv add <package>`. Do not run `pip install`. `uv add` updates `pyproject.toml` and `uv.lock`, so every laptop gets the same packages.

## 2. Data

All raw files go in `data/raw/`. Git ignores this folder.

```bash
mkdir -p data/raw
```

### Pancreas (required, every laptop)

The scIB human pancreas benchmark. 16,382 cells, 19,093 genes, nine protocols in `obs["tech"]`, cell types in `obs["celltype"]`, raw counts in `layers["counts"]`. One client per `tech` value. This is the smoke test dataset.

```bash
curl -L -o data/raw/pancreas.h5ad https://figshare.com/ndownloader/files/24539828
```

### Human Lung Cell Atlas core (Track A, lead's laptop only)

584,944 cells, 28,024 genes, 14 source datasets in `obs["dataset"]`, labels in `obs["ann_level_3"]`. The h5ad file is 5.9 GB. One person downloads it. `make_partitions.py` then writes one small file per client to `data/`, and those files go to the team.

```bash
curl -L -o data/raw/hlca_core.h5ad https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad
```

The URL above came from the CELLxGENE curation API on 2026-11-04 (dataset id `066943a2-fdac-4b29-b348-40cede398e4e`). CELLxGENE can change asset URLs when it reprocesses a dataset. If the download fails, get the current URL:

```bash
curl -s https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293 \
  | python -c "import json,sys; [print(d['cell_count'], a['url']) for d in json.load(sys.stdin)['datasets'] for a in d['assets'] if a['filetype']=='H5AD']"
```

Pick the line with `584944` cells.

### cBioPortal (Tracks B and C, no download step)

`fedbio/cbio.py` reads mutation, copy number, and clinical data from the cBioPortal web API. No login is needed. The loader caches each study in `data/cbio/<study_id>/` on first use. Study IDs are listed in the track files.

## 3. Smoke test

```bash
make test
```

This runs the full pipeline on the pancreas data: partitions, local models, FedAvg, pooled model, the result table, the gain plot, and the top genes per cell type. Target run time: under five minutes on a laptop. Outputs go to `results/`.

Then run the Flower app in simulation:

```bash
flwr run .
```

The Flower numbers must match `make test` within one point of macro F1.

## Rules

- Commit to `main` every hour.
- Ask for help after 20 minutes stuck.
- Flower Message API only: `ServerApp`, `ClientApp`, `Grid`, `@app.train()`. Tutorials that use `NumPyClient`, `server_fn`, or `start_simulation` are the old API. Do not follow them.
