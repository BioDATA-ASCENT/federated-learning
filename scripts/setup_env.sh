#!/usr/bin/env bash
# Create the Python environment with uv.
# Usage: bash scripts/setup_env.sh [3.11|3.12]   (default 3.12)
set -euo pipefail

PY="${1:-3.12}"
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"
uv venv --clear --python "$PY"
uv sync --python "$PY"
# Use .venv/bin/python, not `uv run`. `uv run` obeys .python-version and would
# rebuild the environment with the pinned version instead of $PY.
.venv/bin/python -c "import sys, flwr, scanpy, sklearn, lifelines; print(sys.version.split()[0], 'flwr', flwr.__version__)"

echo
echo "Done. Activate with:  source .venv/bin/activate"
