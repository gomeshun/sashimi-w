"""Compatibility import for :mod:`sashimi_w._itamae`."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module('sashimi_w._itamae')
_sys.modules[__name__] = _module
