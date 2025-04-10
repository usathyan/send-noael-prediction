# Makefile for SEND NOAEL Prediction project

.PHONY: help run-backend install check-venv test

# Default target: Show help
default: help

# Check if the virtual environment directory exists
VENV_DIR := .venv
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment '$(VENV_DIR)' not found."; \
		echo "Please create it first using 'uv venv --python python3.x' (replace 3.x) and install dependencies ('make install')."; \
		exit 1; \
	fi

# Install dependencies using uv
install:
	@if [ ! -f "requirements.txt" ]; then \
		echo "requirements.txt not found."; \
		exit 1; \
	fi
	uv pip install -r requirements.txt
	@echo "Dependencies installed using uv."

# Run the backend FastAPI server with auto-reload
run-backend: check-venv
	@echo "Starting backend server (FastAPI/Uvicorn)..."
	$(VENV_DIR)/bin/python -m uvicorn python.api.main:app --reload --host 127.0.0.1 --port 8000

# Run linters and type checker
test: check-venv
	@echo "Running Ruff linter and auto-fixer..."
	$(VENV_DIR)/bin/ruff check --fix .
	@echo "Running MyPy type checker..."
	$(VENV_DIR)/bin/mypy .
	@echo "Testing commands finished."

# Show help message
help:
	@echo "Available commands:"
	@echo "  make install      Install Python dependencies from requirements.txt into .venv using uv"
	@echo "  make run-backend  Run the backend FastAPI server (localhost:8000)"
	@echo "  make test         Run Ruff linter/fixer and MyPy type checker" 