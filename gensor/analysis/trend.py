"""Analyse trends in the logger data."""

import numpy as np
from matplotlib import pyplot as plt

from ..core.timeseries import Timeseries


def trend_analysis(ts: Timeseries, plot: bool = True) -> None:
    time_numeric = np.arange(len(ts.timeseries))

    # Perform linear regression using numpy's polyfit
    # This returns the slope and intercept of the best fit line
    slope, intercept = np.polyfit(time_numeric, ts.timeseries, 1)

    # Print the slope and intercept
    print(f"Slope: {slope}, Intercept: {intercept}")

    if plot:
        # Compute the values of the trend line
        trend_line = intercept + slope * time_numeric

        # Plotting the original series and the trend line
        plt.figure(figsize=(10, 5))
        plt.plot(ts.timeseries.index, ts.timeseries, label="Original Data")
        plt.plot(ts.timeseries.index, trend_line, color="red", label="Trend Line")
        plt.xlabel("Time")
        plt.ylabel("Groundwater Level")
        plt.title("Groundwater Level Trend Analysis")
        plt.legend()
        plt.show()
