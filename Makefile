.ONESHELL:

SHELL := /bin/bash
DATE_ID := $(shell date +"%y.%m.%d")

# Get package name from pwd
PACKAGE_NAME := $(shell basename $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
OPENVINO_DOCKER_IMAGE = "$(USER)/$(shell basename $(CURDIR))"
SOURCE_DIR = source /opt/intel/openvino/bin/setupvars.sh
TEST_CMD = "$(SOURCE_DIR) && pytest -sv ."

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build-image:  ## Build docker image from file.
	docker build --no-cache -t $(OPENVINO_DOCKER_IMAGE) .

build-cached-image:  ## Build cached docker image from file.
	docker build -t $(OPENVINO_DOCKER_IMAGE) .

clean: clean-build clean-docker clean-pyc clean-test ## Remove all build, test, coverage and Python artefacts

clean-docker:  ## Remove docker image
	docker rmi $(OPENVINO_DOCKER_IMAGE) || true

clean-build: ## Remove build artefacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artefacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artefacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## Check style with flake8
	flake8 --max-line-length 90 $(PACKAGE_NAME)

formatter: ## Format style with black
	isort -rc .
	black -l 90 .

tests: clean build-image ## Run tests in docker
	docker run --rm -ti --volume $(CURDIR):/app $(OPENVINO_DOCKER_IMAGE) \
		bash -c $(TEST_CMD)

changelog: ## Generate changelog for current repo
	docker run -it --rm -v "$(pwd)":/usr/local/src/your-app mmphego/git-changelog-generator

coverage: clean build-image ## Check code coverage quickly with the default Python
	docker run --rm -ti --volume $(CURDIR):/app $(OPENVINO_DOCKER_IMAGE) \
		bash -c "$(SOURCE_DIR) && \
			coverage run --source=$(PACKAGE_NAME) -m pytest -sv . && \
			coverage report -m && \
			coverage html"

view-coverage: ## View generated coverage on firefox
	@firefox htmlcov/index.html
