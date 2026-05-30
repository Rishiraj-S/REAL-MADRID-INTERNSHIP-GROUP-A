.PHONY: install install-dev install-notebooks lint typecheck test quality run train

PYTHON ?= $(shell conda run -n pedri which python 2>/dev/null || python3)

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-notebooks:
	$(PYTHON) -m pip install -e ".[notebooks]"

lint:
	ruff check main.py train_models.py src tests

typecheck:
	mypy

test:
	pytest

quality: lint typecheck test

run:
	streamlit run main.py

train:
	python3 train_models.py

