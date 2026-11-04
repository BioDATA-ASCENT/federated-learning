# models

Weights that are committed because the server needs them and they are small.

| File | What |
|---|---|
| `hlca_initial.pt` | The warm start for `apps/hlca_fedavg`, trained on the two initial HLCA datasets |
| `hlca_initial.json` | What those weights fit: gene count, hidden size, label order, hyperparameters, score |

A state dict carries the tensor shapes but not the genes its columns mean. Loaded
against a gene list in a different order it still runs, trains and averages, and the
result means nothing. So check the sidecar before you reuse these weights.

`scripts/partition_hlca.py` chooses the same 2000 genes on every run, so
`make initial-model` reproduces `hlca_initial.pt` bitwise. Regenerate and compare
hashes if you want to verify this copy.

See DEPLOYMENT.md, "Warm start".
