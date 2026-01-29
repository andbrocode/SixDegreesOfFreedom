"""
sixdegrees - A package for 6-DoF seismic data processing

A comprehensive Python package for processing 6-degree-of-freedom (6-DoF) seismic data,
including array-derived rotation computation, backazimuth analysis, and velocity estimation.
"""

from .sixdegrees import sixdegrees
from .seismicarray import seismicarray

__version__ = "0.1.1"
__all__ = ['sixdegrees', 'seismicarray', '__version__']