PYTHON ?= python
VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
    VENV_PIP := $(VENV_DIR)\Scripts\pip.exe
else
    VENV_PYTHON := $(VENV_DIR)/bin/python
    VENV_PIP := $(VENV_DIR)/bin/pip
endif

.PHONY: install lint format test train serve docker-build docker-run

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -e .

lint:
	$(VENV_PYTHON) -m ruff check .

format:
	$(VENV_PYTHON) -m ruff format .
	touch .formatted

test:
	$(VENV_PYTHON) -m pytest -q

train:
	$(VENV_PYTHON) -m asoc_ml.models.train

serve:
	MOCK=true $(VENV_PYTHON) -m uvicorn asoc_ml.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t asoc-ml-pipeline .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env.example asoc-ml-pipeline
