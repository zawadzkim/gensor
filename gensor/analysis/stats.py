"""Module to compute timeseries statistics, similar to pastas.stats.signatures module
and following Heudorfer et al. 2019

To be implemented:

- Structure
 * Flashiness
- Distribution
 * Modality
 * Density
- Shape
 * Scale
 * Slope
"""

import numpy as np

from gensor.core.timeseries import Timeseries


def trend(ts: Timeseries) -> tuple:
    time_numeric = np.arange(len(ts.timeseries))

    # Perform linear regression using numpy's polyfit
    # This returns the slope and intercept of the best fit line
    slope, intercept = np.polyfit(time_numeric, ts.timeseries, 1)

    return slope, intercept
