# Configuration
PROGEN_REPO = https://github.com/Profluent-AI/progen3
THIRD_PARTY = third_party
PROGEN_DIR = $(THIRD_PARTY)/progen3

# Zenodo (data and checkpoints)
TOKEN = eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhmN2ZkYTg2LTNjNTAtNGRiMy1hZTAxLTVlZmNkNTE2NDUzOCIsImRhdGEiOnt9LCJyYW5kb20iOiIzNmQ5ZGY5MGZkZmQyM2JkNTMxYjA3NjE5NWRmYmRiOSJ9.4Lc51rO-bNVWFLJiwIbKmcejJcRdt8vM3YRm2WfQGSZLTNTWQv2GcHUqoqW8XX6PSjidZxjW4qpPIreFo0tfdQ
ZENODO_ID = 18403586
API_URL = https://zenodo.org/api/records/$(ZENODO_ID)/files

# Tooling Check
UV := $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)
PIXI := $(shell command -v pixi 2> /dev/null || echo $(HOME)/.pixi/bin/pixi)

.PHONY: all setup_progen install_managers install_deps verify download_zenodo download_checkpoints download_results clean

all: setup_progen install_managers install_deps verify download_zenodo
download_zenodo: download_checkpoints download_results

install_managers:
	@# Check if UV exists, otherwise install it
	@if [ ! -x "$(UV)" ]; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@# Check if PIXI exists, otherwise install it
	@if [ ! -x "$(PIXI)" ]; then \
		echo "Installing pixi..."; \
		curl -fsSL https://pixi.sh/install.sh | bash; \
	fi
	. $(HOME)/.profile || . $(HOME)/.bashrc || . $(HOME)/.zshrc;

install_deps:
	@echo "--- 1. Installing Pixi Environments ---"
	$(PIXI) install
	
	@echo "--- 2. Installing CUDA extensions (No Build Isolation) ---"
	# Base/GPT Env: Flash-Attn
	$(PIXI) run -e default uv pip install "flash-attn<2.8" --no-build-isolation
	
	# ProGen3 Env: Flash-Attn + Megablocks
	$(PIXI) run -e progen3 uv pip install \
		"flash-attn==2.7.3" \
		"megablocks[gg]==0.7.0" \
		--no-build-isolation

setup_progen:
	@if [ ! -d "$(PROGEN_DIR)" ]; then \
		echo "Cloning ProGen3..."; \
		mkdir -p $(THIRD_PARTY); \
		git clone $(PROGEN_REPO) $(PROGEN_DIR); \
	fi

verify:
	@echo "--- 3. Verifying Hardware & Kernels ---"
	$(PIXI) run -e default check-gpu
	$(PIXI) run -e default setup-kernel plm-steer-gpt
	$(PIXI) run -e progen3 check-gpu
	$(PIXI) run -e progen3 setup-kernel plm-steer-progen3

download_checkpoints:
	@echo "Downloading models checkpoints..."
	curl -L "$(API_URL)/checkpoints.zip/content?token=$(TOKEN)" -o checkpoints.zip
	unzip -q checkpoints.zip -d .
	rm checkpoints.zip
	@echo "Models checkpoints ready."

download_results:
	@echo "Downloading results..."
	curl -L "$(API_URL)/results.zip/content?token=$(TOKEN)" -o results.zip
	unzip -q results.zip -d .
	rm results.zip
	@echo "Results ready."

clean:
	rm -rf .pixi
	@echo "Environments cleared. Note: $(PROGEN_DIR) was not removed."
