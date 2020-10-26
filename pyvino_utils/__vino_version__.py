from loguru import logger

try:
    from openvino.inference_engine import get_version
except ModuleNotFoundError:
    logger.warning("OpenVINO is not installed, your mileage may vary!")


def openvino_version_check():
    try:
        version = tuple(map(int, get_version().split(".")))[:2]
        if version != (2, 1):
            logger.warning(
                f"OpenVINO version: {version!r} not compatible with this library, "
                f"expected version: 2.1.xxx"
            )
    except NameError:
        version = '0.0.0'
    return version


__vino_version__ = openvino_version_check()
