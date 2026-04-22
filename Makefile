# Makefile for Healthcare_XAI_CMPE_252
# Use module-based execution to avoid "No module named src" errors.

PYTHON ?= /opt/anaconda3/bin/python
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: help doctor dataio ingest transform train-model train predict evaluate all

help:
	@echo "Available targets:"
	@echo "  make doctor       - Show Python info and verify core imports"
	@echo "  make dataio       - Run data loading smoke script"
	@echo "  make ingest       - Run data ingestion module"
	@echo "  make transform    - Run data transformation module"
	@echo "  make train-model  - Run preprocess model trainer module"
	@echo "  make train        - Run models train module"
	@echo "  make predict      - Run prediction module"
	@echo "  make evaluate     - Run evaluation module"
	@echo "  make all          - Run key pipeline steps (dataio -> ingest -> transform -> train)"
	@echo ""
	@echo "Override interpreter if needed:"
	@echo "  make ingest PYTHON=/path/to/python"

doctor:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -c "import sys; print('Python:', sys.executable); import pandas, numpy; print('Imports OK: pandas, numpy')"

dataio:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.data.dataio

ingest:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.preprocess.data_ingestion

transform:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.preprocess.data_transformation

train-model:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.preprocess.model_trainer

train:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.models.train

predict:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.models.predict

evaluate:
	@cd "$(PROJECT_ROOT)" && $(PYTHON) -m src.evaluate.evaluate

all: dataio ingest transform train
