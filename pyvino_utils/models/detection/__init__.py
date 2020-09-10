# Python standard library
import os
import pkgutil

from . import face_detection, person_detection

__all__ = [module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])]
