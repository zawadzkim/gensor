from __future__ import annotations

from typing import Any

import pandas as pd
import pandera as pa
import pydantic as pyd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gensor.core.base import BaseTimeseries

ts_schema = pa.SeriesSchema(
    float,
    index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), coerce=False),
    coerce=True,
)


class Timeseries(BaseTimeseries):
    """Timeseries of groundwater sensor data.

    Attributes:
        ts (pd.Series): The timeseries data.
        variable (Literal['temperature', 'pressure', 'conductivity', 'flux']):
            The type of the measurement.
        unit (Literal['degC', 'mmH2O', 'mS/cm', 'm/s']): The unit of
            the measurement.
        sensor (str): The serial number of the sensor.
        sensor_alt (float): Altitude of the sensor (ncessary to compute groundwater levels).
    """

    model_config = pyd.ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    sensor: str | None = None
    sensor_alt: float | None = None

    def __eq__(self, other: object) -> bool:
        """Check equality based on location, sensor, variable, unit and sensor_alt."""
        if not isinstance(other, Timeseries):
            return NotImplemented

        if not super().__eq__(other):
            return False

        return self.sensor == other.sensor and self.sensor_alt == other.sensor_alt

    def plot(
        self,
        include_outliers: bool = False,
        ax: Axes | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the timeseries data.

        Parameters:
            include_outliers (bool): Whether to include outliers in the plot.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
                If None, a new figure and axes are created.
            plot_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.plot() method to customize the plot.
            legend_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.legend() to customize the legend.

        Returns:
            (fig, ax): Matplotlib figure and axes to allow further customization.
        """
        fig, ax = super().plot(
            include_outliers=include_outliers,
            ax=ax,
            plot_kwargs=plot_kwargs,
            legend_kwargs=legend_kwargs,
        )

        ax.set_title(f"{self.variable.capitalize()} at {self.location} ({self.sensor})")

        return fig, ax
