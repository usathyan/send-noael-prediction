# Makefile for SEND NOAEL Prediction (TxGemma Demo)

.DEFAULT_GOAL := help

# --- Configuration ---
PYTHON := python3
UV := uv
APP_MODULE := src.api.main:app
HOST := 127.0.0.1
PORT := 8000

# Check if running in a uv venv
ifeq ($(findstring .venv,$(VIRTUAL_ENV)),.venv)
    RUN_PREFIX := $(UV) run
else
    RUN_PREFIX := 
endif

# --- Targets ---

install: ## Create virtual env and install dependencies using uv
	@echo "Creating virtual environment and installing dependencies..."
	@$(UV) venv --python $(PYTHON) || (echo "Failed to create virtual environment." && exit 1)
	@$(UV) pip install -r requirements.txt || (echo "Failed to install requirements." && exit 1)
	@echo "Installation complete. Activate with: source .venv/bin/activate"

run: ## Run the FastAPI server using uvicorn
	@echo "Starting FastAPI server at http://$(HOST):$(PORT)..."
	@$(RUN_PREFIX) uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --reload

lint: ## Lint the python code using ruff
	@echo "Linting Python code..."
	@$(RUN_PREFIX) ruff check src/

format: ## Format the python code using ruff
	@echo "Formatting Python code..."
	@$(RUN_PREFIX) ruff format src/

clean: ## Remove generated files and directories
	@echo "Cleaning up..."
	@rm -rf .venv uploaded_studies/__pycache__ src/**/__pycache__ .ruff_cache
	@find . -name ".DS_Store" -delete

help: ## Display this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: install run lint format clean help 