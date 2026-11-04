PY ?= 3.12

.PHONY: env data data-hlca test clean-results

env:            ## create .venv with uv and install packages
	bash scripts/setup_env.sh $(PY)

data:           ## download the pancreas data (every laptop)
	bash scripts/download_data.sh

data-hlca:      ## download pancreas and HLCA core (lead's laptop, 5.87 GB)
	bash scripts/download_data.sh --hlca

test:           ## full pipeline on the pancreas data (target: under five minutes)
	@echo "make test: pipeline not written yet. See Progress.md items 5 to 12."
	@exit 1

clean-results:
	rm -rf results/
