# Quickstart

Two steps: install the environment and download the data. All data are real and public. Nothing here is synthetic.

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
make env            # runs scripts/setup_env.sh: uv venv --python 3.12, then uv sync
source .venv/bin/activate
```

`make env PY=3.11` selects Python 3.11. The script does the same as these three commands: `uv venv --python 3.12`, `uv sync`, and an import check.

`uv venv --python 3.12` downloads Python 3.12 if your machine does not have it. Use `--python 3.11` if you prefer 3.11. `uv sync` installs the packages listed in `pyproject.toml` into `.venv`.

Check the install:

```bash
python -c "import flwr, scanpy, sklearn; print(flwr.__version__)"
```

Expected output: `1.42.0`.

To add a package, run `uv add <package>`. Do not run `pip install`. `uv add` updates `pyproject.toml` and `uv.lock`, so every laptop gets the same packages.

## 2. Data

All raw files go in `data/raw/`. Git ignores this folder. One script downloads everything:

```bash
make data           # pancreas only, every laptop
make data-hlca      # pancreas and HLCA core, lead's laptop only, 5.87 GB
```

Run `make env` first. The script (`scripts/download_data.sh`) calls `.venv/bin/python`.

The script resumes partial downloads, skips files that exist, checks the md5 of the pancreas file, and opens each file with anndata. It downloads to `<file>.part` and renames the file, so a failed download never leaves a file that the next run reports as complete. The manual commands follow.

### Pancreas (required, every laptop)

The scIB human pancreas benchmark. 16,382 cells, 19,093 genes, nine protocols in `obs["tech"]`, cell types in `obs["celltype"]`, raw counts in `layers["counts"]`. One client per `tech` value. The md5 of the file is `5aa155652b998c90395a0c9bd731b4b3`.

Use the host `ndownloader.figshare.com`. The address `https://figshare.com/ndownloader/files/24539828` answers HTTP 202 with an empty body, and `curl --fail` accepts it. That writes a file of zero bytes.

```bash
curl -L -o data/raw/pancreas.h5ad https://ndownloader.figshare.com/files/24539828
```

### Human Lung Cell Atlas core (Track A, lead's laptop only)

584,944 cells, 27,402 genes, 14 source datasets in `obs["dataset"]`, labels in `obs["ann_level_3"]`. The h5ad file is 5.87 GB. One person downloads it. That person then writes one small file per client to `data/` and shares those files with the team.

```bash
curl -L -o data/raw/hlca_core.h5ad https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad
```

The URL above came from the CELLxGENE curation API on 2026-11-03 (dataset id `066943a2-fdac-4b29-b348-40cede398e4e`). CELLxGENE can change asset URLs when it reprocesses a dataset. If the download fails, get the current URL:

```bash
curl -s https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293 \
  | python -c "import json,sys; [print(d['cell_count'], a['url']) for d in json.load(sys.stdin)['datasets'] for a in d['assets'] if a['filetype']=='H5AD']"
```

Pick the line with `584944` cells.

### cBioPortal (Tracks B and C, no download step)

The cBioPortal web API serves mutation, copy number, and clinical data. No login is needed, so there is no download step. Study IDs are listed in the track files.

## Rules

- Commit to `main` every hour.
- Ask for help after 20 minutes stuck.
- Flower Message API only: `ServerApp`, `ClientApp`, `Grid`, `@app.train()`. Tutorials that use `NumPyClient`, `server_fn`, or `start_simulation` are the old API. Do not follow them.
