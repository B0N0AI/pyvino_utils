# -*- coding: utf-8 -*-

"""Top-level package for pyvino-utils."""

__author__ = """Mpho Mphego"""
__email__ = 'mpho112@gmail.com'

from loguru import logger

class PyvinoUtils:
    def __init__(self, log_level="INFO", **kwargs):
        self.logger = logger
        self.logger.level(log_level.upper())

    # Code goes here
