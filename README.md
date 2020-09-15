# PyVINO-Utils

[![Build Status](https://img.shields.io/travis/mmphego/pyvino_utils.svg)](https://travis-ci.com/mmphego/pyvino_utils)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/43713e0b78f547e8912ff05c9350cffb)](https://app.codacy.com/app/mmphego/pyvino_utils?utm_source=github.com&utm_medium=referral&utm_content=mmphego/pyvino_utils&utm_campaign=Badge_Grade_Dashboard)
[![Python](https://img.shields.io/badge/Python-3.6%2B-red.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub](https://img.shields.io/github/license/mmphego/pyvino_utils.svg)](LICENSE)

Simplified OpenVINO models Python implementation.

# Installation

To install `pyvino-utils`, run this command in your terminal:

```python
    pip install git+https://github.com/mmphego/pyvino_utils.git
    
```

**NOTE:** This assumes that the [Intel OpenVINO Toolkit version: 2020.2.120](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) is installed and other dependencies. 
**[Preferred]** Alternatively clone the repository and build the docker image. See [Usage](#usage)

# Usage

```bash
git clone git://github.com/mmphego/pyvino_utils
make build-image
docker run --rm -ti "$USER/$(basename $PWD)" bash -c \
    "source /opt/intel/openvino/bin/setupvars.sh && \
    python -c "import pyvino_utils; print(f'PyVINO-utils version: {pyvino_utils.__version__}')"
    "
```

# Examples

See usage examples:

- [Face Detection](examples/age_gender_recognition)
- [Age Gender Recognition](examples/age_gender_recognition)


# Feedback

Feel free to fork it or send me PR to improve it.

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [mmphego/cookiecutter-python-package](https://github.com/mmphego/cookiecutter-python-package) project template.
