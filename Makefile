.PHONY: install install-all hooks format lint test coverage check build clean \
        download logistic boosted study calibration imbalance features ablation

install:
	poetry install

install-all:
	poetry install --with boost,neural,imbalance,explain,notebooks

hooks:
	poetry run pre-commit install

format:
	poetry run ruff check src tests --fix
	poetry run ruff format src tests

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests
	poetry run mypy src

test:
	poetry run pytest

coverage:
	poetry run pytest --cov=scania_aps --cov-report=term-missing

# What CI runs, in one command.
check: lint test

build:
	poetry build

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

download:
	poetry run scania-aps download

logistic:
	poetry run scania-aps train-logistic --trials 36

boosted:
	poetry run scania-aps train-boosted

study:
	poetry run scania-aps run-study --profile full

calibration:
	poetry run scania-aps study-calibration --model xgboost

imbalance:
	poetry run scania-aps study-imbalance

features:
	poetry run scania-aps study-features

ablation:
	poetry run scania-aps study-ablation
