# Python standard library
import os
import pkgutil

__all__ = [module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])]

# Hackery!!!
# FYI -> Skating on thin ice, literally.
# Be careful with exec (see: https://lucumr.pocoo.org/2011/2/1/exec-in-python/)

for _module in __all__:
    exec(f"from . import {_module}")
