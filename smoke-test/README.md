# Two-laptop Flower smoke test

This app runs one FedAvg round with two clients. Each client adds its configured
`client-delta` to the initial value `0`; with deltas `1` and `3`, the server
should print a final averaged value of `2.0`. It uses no project or patient data.

## Prepare both laptops

Both Laptop A and Laptop B need this repository, Python 3.11 or 3.12, `uv`, and
the project environment. On each laptop, from the repository root:

```bash
make env PY=3.11
source .venv/bin/activate
```

Use a trusted LAN or VPN for this first connectivity test. The commands below
use Flower's **insecure, unencrypted** mode. Do not expose these ports to the
public internet or send sensitive data. Use TLS and authentication before
handling real data.

## Laptop A: SuperLink and client 1

Laptop A hosts the SuperLink and runs the first SuperNode. Find Laptop A's LAN
IP address; below, replace `SERVER_IP` with it.

In one terminal on Laptop A, start the SuperLink:

```bash
flower-superlink --insecure --host 0.0.0.0 --port 8000
```

Allow inbound TCP port `8000` through Laptop A's firewall from Laptop B.

In a second terminal on Laptop A, start its SuperNode:

```bash
flower-supernode --insecure \
  --superlink $SERVER_IP:8000 \
  --host $SERVER_IP \
  --port 9094 \
  --node-config "client-delta=1"
```

## Laptop B: client 2

Find Laptop B's LAN IP address; replace `SERVER_IP` with Laptop A's LAN IP and
`CLIENT_IP` with Laptop B's LAN IP. On Laptop B, start its SuperNode:

```bash
flower-supernode --insecure \
  --superlink SERVER_IP:8000 \
  --host CLIENT_IP \
  --port 9094 \
  --node-config "client-delta=3"
```

Allow Laptop A to reach Laptop B's SuperNode runtime API on TCP port `9094`.
If either laptop is behind NAT or a restrictive firewall, use a VPN that gives
both laptops mutually reachable addresses. Keep the SuperNode terminal open.

## Submit the app from Laptop A

On Laptop A, add a named SuperLink connection to the Flower config file shown
by `flwr config list`:

```toml
[superlink.lan-smoke]
address = "127.0.0.1:8000"
insecure = true
```

Then, from the repository root with its environment activated, run on Laptop A:

```bash
flwr run ./smoke-test lan-smoke --stream
```

The `flwr run` terminal should show a completed run, including:

```text
Smoke test complete: final averaged value = 2.0
```

Press Ctrl+C in each terminal to stop the processes. A failed or incomplete
run usually means an IP/port is not reachable, a firewall is blocking traffic,
or one of the two SuperNodes is not connected.
