from __future__ import annotations

from collections import defaultdict
from typing import Any, Generic

import pydantic as pyd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gensor.core.base import BaseTimeseries, T
from gensor.db import DatabaseConnection
from gensor.exceptions import IndexOutOfRangeError


class Dataset(pyd.BaseModel, Generic[T]):
    """Store and operate on a collection of Timeseries.

    Attributes:
        timeseries (list[Timeseries]): A list of Timeseries objects.
    """

    timeseries: list[T | None] = pyd.Field(default_factory=list)

    def __iter__(self) -> Any:
        """Allows to iterate directly over the dataset."""
        return iter(self.timeseries)

    def __len__(self) -> int:
        """Gives the number of timeseries in the Dataset."""
        return len(self.timeseries)

    def __repr__(self) -> str:
        return f"Dataset({len(self)})"

    def __getitem__(self, index: int) -> T | None:
        """Retrieve a Timeseries object by its index in the dataset.

        !!! warning
            Using index will return the reference to the timeseries. If you need a copy,
            use .filter() instead of Dataset[index]

        Parameters:
            index (int): The index of the Timeseries to retrieve.

        Returns:
            Timeseries: The Timeseries object at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        try:
            return self.timeseries[index]
        except IndexError:
            raise IndexOutOfRangeError(index, len(self)) from None

    def get_locations(self) -> list:
        """List all unique locations in the dataset."""
        return [ts.location for ts in self.timeseries if ts is not None]

    def add(self, other: T | list[T] | Dataset) -> Dataset:
        """Appends new Timeseries to the Dataset.

        If an equal Timeseries already exists, merge the new data into the existing
        Timeseries, dropping duplicate timestamps.

        Parameters:
            other (Timeseries): The Timeseries object to add.
        """

        # I need to check for BaseTimeseries instance in the add() method, but also
        # type hint VarType T.
        if isinstance(other, list | Dataset):
            for ts in other:
                if isinstance(ts, BaseTimeseries):
                    self._add_single_timeseries(ts)  # type: ignore[arg-type]

        elif isinstance(other, BaseTimeseries):
            self._add_single_timeseries(other)

        return self

    def _add_single_timeseries(self, ts: T) -> None:
        """Adds a single Timeseries to the Dataset or merges if an equal one exists."""
        for i, existing_ts in enumerate(self.timeseries):
            if existing_ts == ts:
                self.timeseries[i] = existing_ts.concatenate(ts)
                return

        self.timeseries.append(ts)

        return

    def filter(
        self,
        location: str | list | None = None,
        variable: str | list | None = None,
        unit: str | list | None = None,
        **kwargs: dict[str, str | list],
    ) -> T | Dataset:
        """Return a Timeseries or a new Dataset filtered by station, sensor,
        and/or variable.

        Parameters:
            location (Optional[str]): The location name.
            variable (Optional[str]): The variable being measured.
            unit (Optional[str]): Unit of the measurement.
            **kwargs (dict): Attributes of subclassed timeseries used for filtering
                (e.g., sensor, method).

        Returns:
            Timeseries | Dataset: A single Timeseries if exactly one match is found,
                                   or a new Dataset if multiple matches are found.
        """

        def matches(ts: T, attr: str, value: dict[str, str | list]) -> bool | None:
            """Check if the Timeseries object has the attribute and if it matches the value."""
            if not hasattr(ts, attr):
                message = f"'{ts.__class__.__name__}' object has no attribute '{attr}'"
                raise AttributeError(message)
            return getattr(ts, attr) in value

        if isinstance(location, str):
            location = [location]
        if isinstance(variable, str):
            variable = [variable]
        if isinstance(unit, str):
            unit = [unit]
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = [value]

        matching_timeseries = [
            ts
            for ts in self.timeseries
            if ts is not None
            and (location is None or ts.location in location)
            and (variable is None or ts.variable in variable)
            and (unit is None or ts.unit in unit)
            and all(matches(ts, attr, value) for attr, value in kwargs.items())
        ]

        if not matching_timeseries:
            return Dataset()

        if len(matching_timeseries) == 1:
            return matching_timeseries[0].model_copy(deep=True)

        return self.model_copy(update={"timeseries": matching_timeseries})

    def to_sql(self, db: DatabaseConnection) -> None:
        """Save the entire timeseries to a SQLite database.

        Parameters:
            db (DatabaseConnection): SQLite database connection object.
        """
        for ts in self.timeseries:
            if ts:
                ts.to_sql(db)
        return

    def plot(
        self,
        include_outliers: bool = False,
        plot_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the timeseries data, grouping by variable type.

        Parameters:
            include_outliers (bool): Whether to include outliers in the plot.
            plot_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.plot() method to customize the plot.
            legend_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.legend() to customize the legend.

        Returns:
            (fig, ax): Matplotlib figure and axes to allow further customization.
        """

        grouped_ts = defaultdict(list)

        for ts in self.timeseries:
            if ts:
                grouped_ts[ts.variable].append(ts)

        num_variables = len(grouped_ts)

        fig, axes = plt.subplots(
            num_variables, 1, figsize=(10, 5 * num_variables), sharex=True
        )

        if num_variables == 1:
            axes = [axes]

        for ax, (variable, ts_list) in zip(axes, grouped_ts.items(), strict=False):
            for ts in ts_list:
                ts.plot(
                    include_outliers=include_outliers,
                    ax=ax,
                    plot_kwargs=plot_kwargs,
                    legend_kwargs=legend_kwargs,
                )

            ax.set_title(f"Timeseries for {variable.capitalize()}")
            ax.set_xlabel("Time")

        fig.tight_layout()
        return fig, axes
