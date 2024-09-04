"""Tools for smoothing the data."""

from matplotlib import pyplot as plt
from pandas import Series
from sklearn.metrics import mean_squared_error

from .dtypes import Timeseries


def smooth_data(
    data: Timeseries,
    window: int = 5,
    method: str = "rolling_mean",
    print_statistics: bool = False,
    inplace: bool = False,
    plot: bool = False,
) -> Series | None:
    """Smooth a time series using a rolling mean or median.

    Args:
        data (pandas.Series): The time series data.
        window (int): The size of the window for the rolling mean or median. Defaults to 5.
        method (str): The method to use for smoothing. Either 'rolling_mean' or 'rolling_median'. Defaults to 'rolling_mean'.

    Returns:
        pandas.Series: The smoothed time series.
    """
    if method == "rolling_mean":
        smoothed_data = data.ts.rolling(window=window, center=True).mean()
    elif method == "rolling_median":
        smoothed_data = data.ts.rolling(window=window, center=True).median()
    else:
        raise NotImplementedError()

    valid_indices = smoothed_data.notna()
    original_data_aligned = data.ts[valid_indices]
    smoothed_data_aligned = smoothed_data[valid_indices]

    if print_statistics:
        mse = mean_squared_error(original_data_aligned, smoothed_data_aligned)
        print(f"Mean Squared Error of {method}: {mse:.2f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(
            data.timeseries.index, data.timeseries, label="Original Data", color="black"
        )
        plt.plot(
            smoothed_data.index,
            smoothed_data,
            label=f"Moving Average ({method})",
            color="green",
            linestyle="dotted",
        )

        plt.legend()
        plt.title("Groundwater Level with Moving Average")
        plt.xlabel("Date")
        plt.ylabel("Groundwater Level")
        plt.show()

    if inplace:
        data.ts = smoothed_data
        return None
    else:
        return smoothed_data
