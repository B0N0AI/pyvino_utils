# OpenVINO - Face Detection

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  [![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2020.2](https://img.shields.io/badge/openvino-2020.2-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)|
| Docker (Ubuntu OpenVINO pre-installed): | [mmphego/intel-openvino](https://hub.docker.com/r/mmphego/intel-openvino)|
| Hardware Used: | Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz |
| Device: | CPU |

A simple face detection demo based on the Intel OpenVINO Toolkit which uses the onboard WebCam.

## Installation

NONE, as everything runs in a container.

## Pre-requisites

- PC/Laptop with `Intel Generation 6 CPU` and WebCam. (Preferably running **Linux**)
- `docker`

This examples runs inside a docker container that comes pre-installed with Intel OpenVINO Toolkit and other tools.

See: [Docker docs for installation.](https://docs.docker.com/engine/install/)

## Application Usage

```bash
make run-bootstrap
```

The command above does the following:
- Builds a docker image based on the user and current working directory. 
    eg: `mmphego/face_detection`
- Downloads the models that OpenVINO uses for inference.
- Adds current hostname/username to the list allowed to make connections to the X/graphical server.
- Runs the application inside the pre-built docker image.

### Demo
