# Running the federation across machines

One server runs the Flower **SuperLink** (the coordinator). Each laptop runs a **SuperNode** (a client) and trains on its own data. Only model weights travel between them.

For a non-technical overview to show a new partner site (agreements, data preparation, what leaves the site), see [SITE_ONBOARDING.md](SITE_ONBOARDING.md).

```
 laptop A (node) ──ssh tunnel──┐
 laptop B (node) ──ssh tunnel──┼──>  server $FL_HOST
 laptop C (node) ──ssh tunnel──┘     SuperLink  127.0.0.1:9092
                                     dashboard  127.0.0.1:8050
```

The server's firewall only allows SSH, so nodes connect through an SSH tunnel. Nothing besides SSH is exposed to the internet, and the tunnel encrypts the traffic.

The demo app is [apps/mnist_fedavg](apps/mnist_fedavg). It is a toy FedAvg on MNIST: an MLP (784→128→10) trained with SGD and momentum, ported from [naderAsadi/FedAvg](https://github.com/naderAsadi/FedAvg) to the Flower Message API. Each node downloads MNIST itself (63 MB) and trains on its own slice of it.

## Join as a node (every laptop)

You need macOS or Linux (on Windows, use WSL), and SSH access to the server. Ask the lead for the login.

**1. Set up SSH key login.** You do this once. The join script can't type a password. Set your login and the server address. The values are in the README's [Federation server](README.md#federation-server) section, or ask the lead:

```bash
export FL_USER=<your-login> FL_HOST=<server-address>
ssh-copy-id $FL_USER@$FL_HOST            # enter the password once
ssh $FL_USER@$FL_HOST 'echo ok'          # must print "ok" without asking for a password
```

To keep the settings, add the `export` line to `~/.zshrc` or `~/.bashrc`. The join script reads `FL_USER` and `FL_HOST`, and stops with a reminder if either one is missing.

**2. Get the code and the environment.**

```bash
git clone https://github.com/BioDATA-ASCENT/federated-learning.git   # or, in an existing clone: git pull
cd federated-learning
make env                               # see quickstart.md. Installs torch; on Linux the CUDA build is about 6 GB
```

**3. Join.**

```bash
bash scripts/join_federation.sh                   # uses FL_USER and FL_HOST from step 1
```

You don't need any IDs. The script opens the tunnel, checks that MNIST is downloaded, and starts the node. Leave it running. When the line `SuperNode ID: ...` appears, you are connected. When a run starts, the server tells your node which slice of the data to train on. During a run you will see one line per round, like this:

```
[client 2] round 3: trained on 13500 samples (digits [0, 1, ..., 9]), epoch losses [0.33], 0.1s
[client 2] local val: loss 0.30, acc 0.918
```

Press Ctrl+C to leave. That also closes the tunnel.

**4. Watch.** Open **http://localhost:8050**. The tunnel forwards the server's dashboard to your laptop. It shows global accuracy and loss, each client's training loss and validation accuracy, and the run config. It refreshes every 2 seconds.

### How the data is split

When a run starts, the server takes the N nodes that are online, numbers them 0 to N−1, and sends each node its number and N with every message. Each node cuts MNIST into N slices from the shared seed and trains on its own slice. So the slices never overlap, and no data goes unused.

- **Join before the run starts.** A node that connects during a run is left out until the next run, so the split stays fixed.
- **Leaving during a run is fine.** That node's rounds count as failures, and the other nodes carry on.

## Run the server (lead)

The server is already set up in `~/federated-learning`, with the services in a tmux session called `fl`. To start it from scratch:

```bash
ssh $FL_USER@$FL_HOST                          # the account that runs the server (README.md)
cd ~/federated-learning
bash scripts/server_up.sh                      # SuperLink + dashboard
bash scripts/server_up.sh --local-node         # optional: the server also acts as a node (handy for a 2-node demo with one laptop)
tmux attach -t fl                              # look around; Ctrl+B then D to detach
```

**Start a training run** once the nodes have joined:

```bash
source .venv/bin/activate
flwr run apps/mnist_fedavg aws --stream -c 'min-nodes=4 num-server-rounds=20'
```

- `min-nodes` (default 2) makes the run wait until at least that many nodes are connected. Every node online at that moment takes part. Check who is online with `flwr supernode list aws`. The run log prints the partition assignment, for example `Assigned 3 partitions: node 1541… -> 0, ...`.
- `response-timeout-seconds` sets the deadline for each client train/evaluate phase (default: 120). A node that misses it is logged as `node_timeout` and skipped for that phase; replies received in time are still aggregated. The node is retried in later phases/rounds if it is still connected. This does not forcibly terminate a stuck client process.
- Everything in the `[tool.flwr.app.config]` section of [apps/mnist_fedavg/pyproject.toml](apps/mnist_fedavg/pyproject.toml) can be overridden with `-c`. Useful ones:
  - `non-iid=true`: each client gets only some digits. This is the interesting case for federated learning.
  - `local-epochs=5`: more local training between averaging rounds.
  - `lr`, `momentum`, `batch-size`, `response-timeout-seconds`.
- Each run writes `~/fedbio-runs/mnist_fedavg/<timestamp>/metrics.jsonl` and `final_model.pt` on the server.

**Stop everything:** `tmux kill-session -t fl`. Logs are kept in `~/fl-logs/`.

## Reference results

These are 10 rounds with 2 nodes (this laptop and the server) and the default config:

| Split | Round 1 | Round 5 | Round 10 |
|---|---|---|---|
| IID | 88.96% | 92.62% | 94.38% |
| Non-IID (2 shards per client: one client got digits 4–9, the other 0–4) | 62.43% | 84.69% | 88.05% |
| IID, 3 laptops (18,000 samples each) | 87.61% | 91.83% | 93.42% |

The 3-laptop run is slightly lower per round because each client takes fewer SGD steps per round. More rounds, or `local-epochs=2`, closes the gap.

The IID run matched a local simulation to four decimals, because the runs are deterministic given the seed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Permission denied` or a password prompt from the join script | SSH key login isn't set up. Repeat step 1 |
| `Local port 9092 is already in use` | An old tunnel or node is still running. Stop it, or use `FL_LINK_PORT=9192 FL_DASH_PORT=8150 bash scripts/join_federation.sh ...` and open `localhost:8150` |
| `FileNotFoundError: 'flower-superexec'` | The venv wasn't on PATH. The join script handles this. If you start `flower-supernode` by hand, run `source .venv/bin/activate` first |
| `[server] node ... timed out` | That node did not reply before `response-timeout-seconds`; the current phase proceeds with the replies received. Check that laptop's client output and connection. |
| A round reports `1 failures` with `Node Unavailable`, and the dashboard shows a client error | A node left less than about a minute ago, and the SuperLink still counted it. The round goes on with the remaining nodes. Wait a minute after nodes leave before starting a run |
| The round never starts | Fewer nodes are connected than `min-nodes`. Check the SuperLink window (`tmux attach -t fl`) for `Activated node_id` lines |
| `No evaluate function registered` | The app has no `@app.evaluate()`. Our app has one. With your own app, add one, or set `fraction_evaluate=0.0` |

## How it fits together

- **The nodes never get the code by git.** The server packages `apps/mnist_fedavg` at `flwr run` time and sends it to every node. Nodes still need the Python environment (torch, flwr), because by default a SuperNode runs the app in its own environment and doesn't install the app's packages.
- **The server never sees training data.** It holds only the full MNIST test set, which it uses to score the global model each round ("global test accuracy"). Clients score the model on their own held-out 10% ("federated" curves).
- **Security for this demo:** Flower runs with `--insecure` and without node authentication. The SSH tunnel provides the encryption and access control. For anything beyond a demo, see Flower's TLS and SuperNode authentication docs.
