PY ?= 3.12
HLCA_DIR ?= $(HOME)/fedbio-data/hlca
INITIAL_MODEL ?= $(HOME)/fedbio-data/initial.pt

.PHONY: env data data-hlca partitions initial-model hardware-check

env:            ## create .venv with uv and install packages
	bash scripts/setup_env.sh $(PY)

data:           ## download the pancreas data (every laptop)
	bash scripts/download_data.sh

data-hlca:      ## download pancreas and HLCA core (lead's laptop, 5.87 GB)
	bash scripts/download_data.sh --hlca

partitions:     ## cut the HLCA partitions (lead's laptop, needs data-hlca first)
	.venv/bin/python scripts/partition_hlca.py --output $(HLCA_DIR)

initial-model:  ## train the warm start model on the two initial datasets
	PYTHONPATH=apps/hlca_fedavg .venv/bin/python apps/hlca_fedavg/train_initial.py \
		--data-dir $(HLCA_DIR) --out $(INITIAL_MODEL)

hardware-check: ## benchmark the local CPU with the MNIST FedAvg workload
	.venv/bin/python apps/mnist_fedavg/benchmark_hardware.py
