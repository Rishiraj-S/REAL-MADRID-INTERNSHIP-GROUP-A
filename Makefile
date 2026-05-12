.PHONY: install install-dev install-notebooks lint typecheck test quality run train

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

install-notebooks:
	python -m pip install -e ".[notebooks]"

lint:
	ruff check app.py train_models.py src tests utils

typecheck:
	mypy

test:
	pytest

quality: lint typecheck test

run:
	streamlit run app.py

train:
	python train_models.py

