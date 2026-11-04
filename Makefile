PY ?= 3.12

.PHONY: env data data-hlca hardware-check

env:            ## create .venv with uv and install packages
	bash scripts/setup_env.sh $(PY)

data:           ## download the pancreas data (every laptop)
	bash scripts/download_data.sh

data-hlca:      ## download pancreas and HLCA core (lead's laptop, 5.87 GB)
	bash scripts/download_data.sh --hlca

hardware-check: ## benchmark the local CPU with the MNIST FedAvg workload
	.venv/bin/python apps/mnist_fedavg/benchmark_hardware.py
