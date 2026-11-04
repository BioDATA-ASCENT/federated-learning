#!/usr/bin/env bash
# Join the federation as a node (SuperNode).
#
#   bash scripts/join_federation.sh                      # as user1@100.53.208.149
#   FL_USER=user3 bash scripts/join_federation.sh        # your own login on the server
#   bash scripts/join_federation.sh user3@other-host     # full override (or FL_SERVER=...)
#
# Opens an SSH tunnel to the server (SuperLink port and dashboard), then runs a
# SuperNode in the foreground. Ctrl+C stops the node and closes the tunnel.
# No IDs needed: at the start of each run the server numbers the online nodes
# and tells each one which slice of the data to train on.
set -euo pipefail

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  echo "Partition numbers are no longer needed: the server assigns them. Run: bash scripts/join_federation.sh" >&2
  exit 1
fi
SERVER="${1:-${FL_SERVER:-${FL_USER:-user1}@${FL_HOST:-100.53.208.149}}}"
LINK_PORT="${FL_LINK_PORT:-9092}"       # SuperLink port on the server (and local end of the tunnel)
DASH_PORT="${FL_DASH_PORT:-8050}"       # dashboard port on the server (and local end of the tunnel)

cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/flower-supernode ]]; then
  echo "No .venv found. Run 'make env' first (see quickstart.md)." >&2
  exit 1
fi
# The SuperNode starts flower-superexec by name, so the venv must be on PATH.
export PATH="$PWD/.venv/bin:$PATH"

for p in "$LINK_PORT" "$DASH_PORT"; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Local port $p is already in use (an old tunnel or node?). Stop it, or set FL_LINK_PORT / FL_DASH_PORT." >&2
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >&2 || true
    exit 1
  fi
done

echo "Opening SSH tunnel to $SERVER (SuperLink -> localhost:$LINK_PORT, dashboard -> localhost:$DASH_PORT)"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "$LINK_PORT:127.0.0.1:9092" -L "$DASH_PORT:127.0.0.1:8050" "$SERVER" &
TUNNEL=$!
trap 'echo; echo "Closing tunnel"; kill $TUNNEL 2>/dev/null || true' EXIT

# Wait for the tunnel to come up (or fail).
for _ in $(seq 1 30); do
  if ! kill -0 "$TUNNEL" 2>/dev/null; then
    echo "SSH tunnel failed. Check 'ssh $SERVER' works without a password prompt." >&2
    exit 1
  fi
  if nc -z 127.0.0.1 "$LINK_PORT" 2>/dev/null; then break; fi
  sleep 0.5
done

# MNIST is downloaded on first use; do it now so round 1 does not wait.
(cd apps/mnist_fedavg && python -c "from mnist_fedavg.task import _load_mnist; _load_mnist('~/.cache/fedbio/mnist', True)") \
  && echo "MNIST ready in ~/.cache/fedbio/mnist"

echo "Dashboard: http://localhost:$DASH_PORT"
echo "Starting SuperNode. The server assigns this node's data slice when a run starts.  (Ctrl+C to leave)"
flower-supernode --insecure --superlink "127.0.0.1:$LINK_PORT"
