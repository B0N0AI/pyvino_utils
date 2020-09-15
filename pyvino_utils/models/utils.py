import os
import pathlib
import sys
from collections import namedtuple
from contextlib import suppress

import requests

MODEL_DOWNLOADER = "/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/"
ARGS = namedtuple("args", ("name", "precision", "print_all"))


if not os.path.exists(MODEL_DOWNLOADER):
    # FIXME: Do stuff and log.warning
    pass

sys.path.insert(0, MODEL_DOWNLOADER)


import downloader  # isort:skip


def download_from_url(urls, out_dir):
    # FIXME: Add requests.Session.get to download urls to out_dir
    pass


def model_downloader(model_name, model_precision="FP16", out_dir="models/"):
    ARGS.name = model_name
    ARGS.precision = model_precision.upper()
    ARGS.print_all = None

    models = None
    with suppress(BaseException):
        models = downloader.common.load_models_from_args(downloader, ARGS)

    if models is None:
        # log warning
        return

    urls = []
    for model in models:
        if model.name == model_name and model_precision in model.precisions:
            for file in model.files:
                urls.append(file.source.url)

    download_from_url(urls, out_dir)
