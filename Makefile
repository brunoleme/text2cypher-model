.PHONY: install test lint format clean clean-artifacts check run train evaluate clean-venv create-venv clean-all

ENV ?= dev
CONFIG_PATH ?= src/text2cypher/finetuning/config
CONFIG_FILE := $(CONFIG_PATH)/config.$(ENV).yaml

help:
	@echo "Makefile commands:"
	@echo "  install                      Install dependencies with uv"
	@echo "  test                         Run tests"
	@echo "  lint                         Run code linters"
	@echo "  format                       Auto-format code"
	@echo "  clean                        Remove Python and test cache"
	@echo "  clean-artifacts              Remove checkpoint artifacts"
	@echo "  run                           Run the full pipeline (default: ENV=dev)"
	@echo "  train                         Run only training"
	@echo "  evaluate                      Run only evaluation"
	@echo "  docker-build                  Build Docker image for ECR"
	@echo "  docker-push                   Push Docker image to ECR"
	@echo "  sagemaker-trigger             Trigger SageMaker training job"
	@echo "  sagemaker-deploy-endpoint      Deploy model to SageMaker Endpoint"


ARTIFACTS_DIR := $(shell \
	if [ -f config.yaml ]; then \
		python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['training']['checkpoint_dir'])"; \
	else \
		echo "checkpoints"; \
	fi)

install:
	pip install -e .[dev]

test:
	PYTHONPATH=. ENV=$(ENV) pytest tests/ --cov=src --cov-report=term-missing -s

lint:
	ruff check .
	black . --check
	mypy src/

format:
	black .
	ruff check . --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name ".coverage*" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +


clean-artifacts:
	rm -rf $(ARTIFACTS_DIR)/*

clean-venv:
	rm -rf .venv

create-venv:
	python3.9 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip setuptools

clean-all: clean clean-venv

check: lint test

check-env:
	@test -f $(CONFIG_FILE) || (echo "Missing config file: $(CONFIG_FILE)" && exit 1)

train:
	PYTHONPATH=. ENV=$(ENV) python scripts/train.py

evaluate:
	PYTHONPATH=. ENV=$(ENV) python scripts/evaluate_model.py

run: check-env
	PYTHONPATH=. ENV=$(ENV) python scripts/run_pipeline.py

docker-build:
	docker build -t $(ECR_REPOSITORY_URI):$(IMAGE_TAG) .

docker-push:
	docker push $(ECR_REPOSITORY_URI):$(IMAGE_TAG)

sagemaker-trigger:
	@MODEL_URI=$$(python .github/scripts/trigger_sagemaker.py \
		--image-uri $(ECR_REPOSITORY_URI):$(IMAGE_TAG) \
		--role-arn $(SAGEMAKER_ROLE_ARN) \
		--job-name text2cypher-$(ENV) \
		--env $(ENV)) && \
	echo "MODEL_ARTIFACT_S3_URI=$$MODEL_URI" >> $(GITHUB_ENV)

sagemaker-deploy-endpoint:
	python .github/scripts/deploy_sagemaker_endpoint.py \
		--image-uri $(ECR_REPOSITORY_URI):$(IMAGE_TAG) \
		--role-arn $(SAGEMAKER_ROLE_ARN) \
		--endpoint-name text2cypher-$(ENV)-endpoint \
		--model-data-s3-uri $(MODEL_ARTIFACT_S3_URI) \
		--instance-type ml.m5.large \
		--instance-count 1
