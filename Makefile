.PHONY: install install-all test lint download logistic boosted study calibration imbalance features ablation

install:
	poetry install

install-all:
	poetry install --with boost,neural,imbalance,explain,notebooks

test:
	poetry run pytest

lint:
	poetry run ruff check src tests
	poetry run mypy src

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
