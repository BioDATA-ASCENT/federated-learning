#!/usr/bin/env bash
# Download the raw data into data/raw/.
# Usage:
#   bash scripts/download_data.sh            # pancreas only (every laptop)
#   bash scripts/download_data.sh --hlca     # pancreas and HLCA core (lead's laptop, 5.87 GB)
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

# Use the ndownloader.figshare.com host. https://figshare.com/ndownloader/files/<id>
# answers 202 with an empty body, which curl --fail accepts.
PANCREAS_URL="https://ndownloader.figshare.com/files/24539828"
PANCREAS_MD5="5aa155652b998c90395a0c9bd731b4b3"
HLCA_COLLECTION="6f6d381a-7701-4781-935c-db10d30de293"
HLCA_DATASET_ID="066943a2-fdac-4b29-b348-40cede398e4e"   # HLCA core, 584,944 cells, 27,402 genes
HLCA_URL_CACHED="https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"

fetch () {
  # fetch <url> <dest>. Resumes a partial download. Skips a complete one.
  # Downloads to <dest>.part and renames it, so a failed download never
  # leaves a file that the next run reports as complete.
  local url="$1" dest="$2" code
  if [ -s "$dest" ]; then
    echo "exists: $dest"
    return
  fi
  echo "download: $dest"
  code=$(curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest.part" \
    -w '%{http_code}' "$url")
  if [ "$code" != "200" ] && [ "$code" != "206" ]; then
    echo "download failed: HTTP $code from $url" >&2
    exit 1
  fi
  if [ ! -s "$dest.part" ]; then
    echo "download failed: empty file from $url" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

check_md5 () {
  # check_md5 <file> <expected>. Skips the check if no md5 tool is present.
  local file="$1" want="$2" got
  if command -v md5sum >/dev/null 2>&1; then
    got=$(md5sum "$file" | cut -d' ' -f1)
  elif command -v md5 >/dev/null 2>&1; then
    got=$(md5 -q "$file")
  else
    echo "no md5 tool: skipping the checksum of $file"
    return
  fi
  if [ "$got" != "$want" ]; then
    echo "md5 mismatch for $file: got $got, want $want" >&2
    exit 1
  fi
  echo "md5 ok: $file"
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
check_md5 "$RAW/pancreas.h5ad" "$PANCREAS_MD5"
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
    echo "API lookup failed. Using the cached URL from 2026-11-03."
    url="$HLCA_URL_CACHED"
  fi
  fetch "$url" "$RAW/hlca_core.h5ad"
  check_h5ad "$RAW/hlca_core.h5ad"
fi

echo
du -sh "$RAW"/*.h5ad
