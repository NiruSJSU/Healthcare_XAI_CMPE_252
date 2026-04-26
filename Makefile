PYTHON ?= python

.PHONY: all dataio ingest transform train evaluate clean-results clean-all

#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------

dataio:
	$(PYTHON) -m src.data.dataio

ingest:
	$(PYTHON) -m src.preprocess.data_ingestion

transform:
	$(PYTHON) -m src.preprocess.data_transformation

train-model:
	$(PYTHON) -m src.preprocess.model_trainer

train:
	$(PYTHON) -m src.models.train

predict:
	$(PYTHON) -m src.models.predict

evaluate:
	$(PYTHON) -m src.evaluate.evaluate

all: clean-results dataio ingest transform train evaluate

#---------------------------------------------------
# Cleaning folders
#---------------------------------------------------
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
# Delete all data
clean-data:
	rm -rf Data/Raw/*
	rm -rf Data/Processed/*

# Delete all models, metrics, and visualizations
clean-results:
	rm -rf models/*
	rm -rf results/*
	rm -rf reports/figures/*

# Delete all
clean-all: clean clean-data clean-results