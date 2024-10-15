from __future__ import annotations

from collections import defaultdict
from typing import Any, Self

import pydantic as pyd
from matplotlib import pyplot as plt

from gensor.core.timeseries import Timeseries
from gensor.db import DatabaseConnection
from gensor.exceptions import IndexOutOfRangeError, TimeseriesNotFound


class Dataset(pyd.BaseModel):
    """Class to store a collection of timeseries.

    The Dataset class is used to store a collection of Timeseries objects. It
    is meant to be created when the van Essen CSV file is parsed.

    Attributes:
        timeseries (list[Timeseries]): A list of Timeseries objects.

    Methods:
        __iter__: Returns timeseries when iterated over.
        __len__: Gives the number of timeseries in the Dataset.
        get_stations: List all unique locations in the dataset.
        add: Appends a new series to the Dataset or merges series if
            an equal one exists.
        align: Aligns the timeseries to a common time axis.
        plot: Plots the timeseries data.
    """

    timeseries: list[Timeseries | None] = pyd.Field(default_factory=list)

    def __iter__(self) -> Any:
        """Allows to iterate directly over the dataset."""
        return iter(self.timeseries)

    def __len__(self) -> int:
        """Gives the number of timeseries in the Dataset."""
        return len(self.timeseries)

    def __repr__(self) -> str:
        return f"Dataset({len(self)})"

    def __getitem__(self, index: int) -> Timeseries | None:
        """Retrieve a Timeseries object by its index in the dataset.

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

    def get_stations(self) -> list:
        """List all unique locations in the dataset."""
        return [ts.location for ts in self.timeseries if ts is not None]

    def add(self, other: Timeseries | list[Timeseries] | Self) -> None:
        """Appends a new series to the Dataset or merges series if an equal
        one exists.

        If a Timeseries with the same location, sensor, and variable already
        exists, merge the new data into the existing Timeseries, dropping
        duplicate timestamps.

        Parameters:
            other (Timeseries): The Timeseries object to add.
        """
        if isinstance(other, list):
            for ts in other:
                if isinstance(ts, Timeseries):
                    self._add_single_timeseries(ts)
        elif isinstance(other, Dataset):
            for ts in other.timeseries:  # type: ignore[assignment]
                if isinstance(ts, Timeseries):
                    self._add_single_timeseries(ts)
        elif isinstance(other, Timeseries):
            self._add_single_timeseries(other)

        return

    def _add_single_timeseries(self, ts: Timeseries) -> None:
        """Adds a single Timeseries to the Dataset or merges if an equal one exists."""
        for i, existing_ts in enumerate(self.timeseries):
            if existing_ts == ts:
                self.timeseries[i] = existing_ts.concatenate(ts)
                return

        self.timeseries.append(ts)

        return

    def filter(
        self,
        stations: str | list | None = None,
        sensors: str | list | None = None,
        variables: str | list | None = None,
    ) -> Timeseries | Dataset:
        """Return a Timeseries or a new Dataset filtered by station, sensor,
        and/or variable.

        Parameters:
            stations (Optional[str]): The location of the station.
            sensors (Optional[str]): The sensor identifier.
            variables (Optional[str]): The variable being measured.

        Returns:
            Timeseries or Dataset: A single Timeseries if exactly one match is found,
                                   or a new Dataset if multiple matches are found.
        """

        if isinstance(stations, str):
            stations = [stations]

        if isinstance(sensors, str):
            sensors = [sensors]

        if isinstance(variables, str):
            variables = [variables]

        matching_timeseries = [
            ts
            for ts in self.timeseries
            if ts is not None
            if (stations is None or ts.location in stations)
            and (sensors is None or ts.sensor in sensors)
            and (variables is None or ts.variable in variables)
        ]

        if not matching_timeseries:
            raise TimeseriesNotFound()

        if len(matching_timeseries) == 1:
            return matching_timeseries[0]

        return self.model_copy(update={"timeseries": matching_timeseries})

    def to_sql(self, db: DatabaseConnection) -> None:
        for ts in self.timeseries:
            if ts:
                ts.to_sql(db)
        return

    def plot(self, include_outliers: bool = False) -> None:
        """Plots the timeseries data, grouping by variable type.

        Args:
            include_outliers (bool): Whether to include outliers in the plot.
        """
        # Group timeseries by variable
        grouped_ts = defaultdict(list)
        for ts in self.timeseries:
            if ts:
                grouped_ts[ts.variable].append(ts)

        # Create a plot for each group of timeseries with the same variable
        for variable, ts_list in grouped_ts.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            for ts in ts_list:
                ts.plot(include_outliers=include_outliers, ax=ax)

            ax.set_title(f"Timeseries for {variable.capitalize()}")
            plt.show()

        return
