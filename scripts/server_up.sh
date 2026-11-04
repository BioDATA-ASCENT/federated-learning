#!/usr/bin/env bash
# Start the federation server (run ON the server): SuperLink + live dashboard,
# each in a window of the tmux session "fl". Both listen on 127.0.0.1 only;
# nodes reach them through SSH tunnels (scripts/join_federation.sh).
#
#   bash scripts/server_up.sh            # start (no-op if already running)
#   bash scripts/server_up.sh --local-node   # also run a node on the server itself
#   tmux attach -t fl                    # watch; Ctrl+B then a window number to switch
#   tmux kill-session -t fl              # stop everything
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
LOGS="$HOME/fl-logs"
mkdir -p "$LOGS" "$HOME/fedbio-runs/mnist_fedavg"

if [[ ! -x .venv/bin/flower-superlink ]]; then
  echo "No .venv found. Run: bash scripts/setup_env.sh 3.12" >&2
  exit 1
fi
ENV="export PATH=$REPO/.venv/bin:\$PATH; cd $REPO"

if ! tmux has-session -t fl 2>/dev/null; then
  tmux new-session -d -s fl -n superlink \
    "$ENV; flower-superlink --insecure --host 127.0.0.1 --port 9092 --database $LOGS/superlink.db 2>&1 | tee -a $LOGS/superlink.log"
  tmux new-window -t fl -n dashboard \
    "$ENV; python apps/mnist_fedavg/dashboard.py --host 127.0.0.1 --port 8050 2>&1 | tee -a $LOGS/dashboard.log"
  echo "Started tmux session 'fl' (superlink, dashboard)."
else
  echo "tmux session 'fl' already running."
fi

if [[ "${1:-}" == "--local-node" ]]; then
  tmux kill-window -t fl:node-server 2>/dev/null || true
  tmux new-window -t fl -n node-server \
    "$ENV; flower-supernode --insecure --superlink 127.0.0.1:9092 --port 9095 2>&1 | tee -a $LOGS/node-server.log"
  echo "Started a local node on the server."
fi

# flwr CLI connection named "aws", used by: flwr run apps/mnist_fedavg aws --stream
CFG="$HOME/.flwr/config.toml"
mkdir -p "$HOME/.flwr"
if [[ ! -f "$CFG" ]]; then
  printf '[superlink]\ndefault = "aws"\n\n[superlink.aws]\naddress = "127.0.0.1:9092"\ninsecure = true\n' > "$CFG"
elif ! grep -q '^\[superlink.aws\]' "$CFG"; then
  printf '\n[superlink.aws]\naddress = "127.0.0.1:9092"\ninsecure = true\n' >> "$CFG"
fi
tmux list-windows -t fl
