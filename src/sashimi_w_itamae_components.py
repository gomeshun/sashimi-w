"""Compatibility import for :mod:`sashimi_w._itamae_components`."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module('sashimi_w._itamae_components')
_sys.modules[__name__] = _module
