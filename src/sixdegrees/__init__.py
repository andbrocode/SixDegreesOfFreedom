"""
sixdegrees - A package for 6-DoF seismic data processing

A comprehensive Python package for processing 6-degree-of-freedom (6-DoF) seismic data,
including array-derived rotation computation, backazimuth analysis, and velocity estimation.
"""

from .sixdegrees import sixdegrees
from .seismicarray import seismicarray
from .plots import (
    plot_backazimuth_results,
    plot_dispersion_curves,
    plot_dispersion_traces,
    plot_velocities,
    plot_waveform_cc,
)

__version__ = "1.0.3"
__all__ = [
    "sixdegrees",
    "seismicarray",
    "plot_backazimuth_results",
    "plot_dispersion_curves",
    "plot_dispersion_traces",
    "plot_velocities",
    "plot_waveform_cc",
    "__version__",
]
