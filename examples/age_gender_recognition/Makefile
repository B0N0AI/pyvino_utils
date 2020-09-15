.ONESHELL:

SHELL := /bin/bash
# Name of docker image to be built.
OPENVINO_DOCKER_IMAGE = "$(USER)/$(shell basename $(CURDIR))"
# Get package name from pwd
SOURCE_DIR = source /opt/intel/openvino/bin/setupvars.sh

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

clean: clean-build clean-docker clean-pyc clean-test ## Remove all build, test, coverage and Python artefacts

clean-docker:  ## Remove docker image
	docker rmi $(OPENVINO_DOCKER_IMAGE)

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

formatter: ## Format style with black
	isort -rc .
	black -l 90 .

lint: ## Check style with flake8
	flake8 --max-line-length 90 .

build:  ## Build docker image from file.
	docker build --no-cache -t $(OPENVINO_DOCKER_IMAGE) .

build-cached:  ## Build cached docker image from file.
	docker build -t $(OPENVINO_DOCKER_IMAGE) .

download-models: build  ## Download models for use
	docker run --rm -ti --volume "$(CURDIR)":/app --env DISPLAY=$(DISPLAY) \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/snd \
	--device /dev/video0 $(OPENVINO_DOCKER_IMAGE) \
	bash -c "/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --list models.lst -o models"

run-bootstrap: build-cached download-models ## Run bootstrap example inside the container.
	xhost +;
	docker run --rm -ti --volume "$(CURDIR)":/app --env DISPLAY=$(DISPLAY) \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/snd \
	--device /dev/video0 $(OPENVINO_DOCKER_IMAGE) \
	bash -c "source /opt/intel/openvino/bin/setupvars.sh && \
			python age_gender_recognition.py \
				--model models/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013 \
				--input cam \
				--show-bbox"

run:  ## Run example
	xhost +;
	docker run --rm -ti --volume "$(CURDIR)":/app --env DISPLAY=$(DISPLAY) \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/snd \
	--device /dev/video0 $(OPENVINO_DOCKER_IMAGE) \
	bash -c "source /opt/intel/openvino/bin/setupvars.sh && \
			python age_gender_recognition.py \
				--model models/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013 \
				--input cam \
				--show-bbox"
