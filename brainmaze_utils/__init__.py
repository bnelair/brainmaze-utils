"""
=======================================
Brainmaze utils (:mod:`brainmaze_utils`)
=======================================

"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("brainmaze-utils")
except PackageNotFoundError:
    __version__ = "0.0.0"