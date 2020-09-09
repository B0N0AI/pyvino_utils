# flake8: noqa


class InvalidModel(Exception):
    """Model loaded unknown. Redownload model from model_downloader or re-optimiser the model."""


class InvalidImageArray(Exception):
    """Image array parsed invalid."""
