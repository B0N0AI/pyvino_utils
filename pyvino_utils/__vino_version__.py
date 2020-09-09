from loguru import logger

from openvino.inference_engine import get_version


def openvino_version_check():
    version = tuple(map(int, get_version().split(".")))[:2]
    if version != (2, 1):
        logger.warning(
            f"OpenVINO version: {version!r} not compatible with this library, "
            f"expected version: 2.1.xxx"
        )
    return version


__vino_version__ = openvino_version_check()
