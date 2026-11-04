#!/usr/bin/env bash
# Download the raw data into data/raw/.
# Usage:
#   bash scripts/download_data.sh            # pancreas only (every laptop)
#   bash scripts/download_data.sh --hlca     # pancreas and HLCA core (lead's laptop, 5.9 GB)
set -euo pipefail

cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

# Use .venv/bin/python, not `uv run`. `uv run` obeys .python-version and would
# rebuild a 3.11 environment with the pinned version.
PYBIN=.venv/bin/python
if [ ! -x "$PYBIN" ]; then
  echo "No environment at .venv. Run 'make env' first." >&2
  exit 1
fi

PANCREAS_URL="https://figshare.com/ndownloader/files/24539828"
HLCA_COLLECTION="6f6d381a-7701-4781-935c-db10d30de293"
HLCA_DATASET_ID="066943a2-fdac-4b29-b348-40cede398e4e"   # HLCA core, 584,944 cells
HLCA_URL_CACHED="https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"

fetch () {
  # fetch <url> <dest>. Resumes a partial download. Skips a complete one.
  local url="$1" dest="$2"
  if [ -s "$dest" ] && [ ! -f "$dest.part" ]; then
    echo "exists: $dest"
    return
  fi
  echo "download: $dest"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
}

check_h5ad () {
  "$PYBIN" - "$1" <<'PY'
import sys, anndata as ad
a = ad.read_h5ad(sys.argv[1], backed="r")
print(f"ok: {sys.argv[1]}  cells={a.n_obs:,}  genes={a.n_vars:,}")
PY
}

# 1. Pancreas (scIB benchmark, 16,382 cells, 9 protocols)
fetch "$PANCREAS_URL" "$RAW/pancreas.h5ad"
check_h5ad "$RAW/pancreas.h5ad"

# 2. HLCA core (optional)
if [ "${1:-}" = "--hlca" ]; then
  echo "Looking up the current HLCA core URL from the CELLxGENE API."
  url=$(curl -s "https://api.cellxgene.cziscience.com/curation/v1/collections/$HLCA_COLLECTION" \
    | "$PYBIN" -c "
import json, sys
col = json.load(sys.stdin)
for d in col['datasets']:
    if d['dataset_id'] == '$HLCA_DATASET_ID':
        for a in d['assets']:
            if a['filetype'] == 'H5AD':
                print(a['url'])
" 2>/dev/null || true)
  if [ -z "$url" ]; then
    echo "API lookup failed. Using the cached URL from 2026-11-04."
    url="$HLCA_URL_CACHED"
  fi
  fetch "$url" "$RAW/hlca_core.h5ad"
  check_h5ad "$RAW/hlca_core.h5ad"
fi

echo
du -sh "$RAW"/*.h5ad
