.ONESHELL:

SHELL := /bin/bash
DATE_ID := $(shell date +"%y.%m.%d")

# Get package name from pwd
PACKAGE_NAME := $(shell basename $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
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


.PHONY: clean clean-test clean-pyc clean-build docs help changelog

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build-image:
	docker build -t "$(USER)/$(shell basename $(CURDIR))" .

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 --max-line-length 90 $(PACKAGE_NAME) tests

formatter: ## Format style with black
	isort -rc .
	black -l 90 .

test: clean build-image ## run tests in docker
	docker run --rm -ti --volume $(CURDIR):/app $(USER)/$(shell basename $(CURDIR)) \
		bash -c $(TEST_CMD)

changelog: ## Generate changelog for current repo
	docker run -it --rm -v "$(pwd)":/usr/local/src/your-app mmphego/github-changelog

coverage: clean build-image ## check code coverage quickly with the default Python
	docker run --rm -ti --volume $(CURDIR):/app $(USER)/$(shell basename $(CURDIR)) \
		bash -c "$(SOURCE_DIR) && \
			coverage run --source=$(PACKAGE_NAME) -m pytest -sv . && \
			coverage report -m && \
			coverage html"

view-coverage:
	@firefox htmlcov/index.html
