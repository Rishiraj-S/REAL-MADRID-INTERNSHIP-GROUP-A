.PHONY: install install-dev install-notebooks lint typecheck test quality run train

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

install-notebooks:
	python3 -m pip install -e ".[notebooks]"

lint:
	ruff check main.py train_models.py src tests utils

typecheck:
	mypy

test:
	pytest

quality: lint typecheck test

run:
	streamlit run main.py

train:
	python3 train_models.py

