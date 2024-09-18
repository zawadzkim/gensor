"""
!!! warning

    Whenever Timeseries objects are created via read_from_csv and use a parser (e.g.,
    'vanessen'), the timestamps are localized and converted to UTC. Therefore, if the
    user creates his own timeseries outside the read_from_csv, they should ensure that
    the timestamps are in UTC format.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, Self

import pandas as pd
import pandera as pa
import pydantic as pyd
from matplotlib import pyplot as plt
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import DatabaseConnection
from .exceptions import IndexOutOfRangeError, TimeseriesNotFound, TimeseriesUnequal
from .preprocessing import OutlierDetection, Transform

ts_schema = pa.SeriesSchema(
    float,
    index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), coerce=False),
    coerce=True,
)

VARIABLE_TYPES_AND_UNITS = {
    "temperature": ["degc"],
    "pressure": ["cmh2o", "mmh2o"],
    "conductivity": ["ms/cm"],
    "flux": ["m/s"],
    "head": ["m asl"],
    "depth": ["m"],
}


class Timeseries(pyd.BaseModel):
    """Timeseries from a sensor including measurement metadata.

    This is class for any sensor timeseries. The basic required attributes are
    just the ts, variable and unit. SensorInfo object is created from the
    relevant kwargs if they are passed.

    Timeseries represents a series of measurements of a single variable, from a
    single sensor with unique timestamps.

    TODO: Perhaps it would be cool to implement kind of a tracking of which
    analyses were performed on the timeseries?

    Attributes:
        ts (pd.Series): The timeseries data.
        variable (Literal['temperature', 'pressure', 'conductivity', 'flux']):
            The type of the measurement.
        unit (Literal['degC', 'mmH2O', 'mS/cm', 'm/s']): The unit of
            the measurement.
        sensor (SensorInfo): The serial number of the sensor.
        analysis (Analysis): An object containing details of analysis done
            on the timeseries.

    Methods:
        validate_ts: if the pd.Series is not exactly what is required, coerce.
    """

    model_config = pyd.ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    ts: pd.Series = pyd.Field(repr=False)
    variable: Literal[
        "temperature", "pressure", "conductivity", "flux", "head", "depth"
    ]
    unit: Literal["degc", "cmh2o", "ms/cm", "m/s", "m asl", "m"]
    location: str | None = None
    sensor: str | None = None
    sensor_alt: float | None = None
    outliers: pd.Series | None = pyd.Field(default=None, repr=False)
    transformation: Any = pyd.Field(default=None, repr=False)

    @pyd.computed_field()  # type: ignore[prop-decorator]
    @property
    def start(self) -> pd.Timestamp | Any:
        return self.ts.index.min()

    @pyd.computed_field()  # type: ignore[prop-decorator]
    @property
    def end(self) -> pd.Timestamp | Any:
        return self.ts.index.max()

    def __eq__(self, other: object) -> bool:
        """Check equality based on location, sensor, and variable."""
        if not isinstance(other, Timeseries):
            return NotImplemented

        return (
            self.variable == other.variable
            and self.unit == other.unit
            and self.location == other.location
            and self.sensor == other.sensor
            and self.start == other.start
        )

    def __getattr__(self, attr: Any) -> Any:
        """Delegate attribute access to the underlying pandas Series if it exists."""

        error_message = f"'{self.__class__.__name__}' object has no attribute '{attr}'"

        if hasattr(self.ts, attr):
            # Return a function to call on the `ts` if it's a method, otherwise return the attribute
            ts_attr = getattr(self.ts, attr)
            if callable(ts_attr):

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = ts_attr(*args, **kwargs)
                    # If the result is a Series, return a new Timeseries; otherwise, return the result
                    if isinstance(result, pd.Series):
                        return self.model_copy(update={"ts": result}, deep=True)
                    return result

                return wrapper
            else:
                return ts_attr
        raise AttributeError(error_message)

    @pyd.field_validator("ts")
    def validate_ts(cls, v: pd.Series) -> pd.Series:
        return ts_schema.validate(v)

    @pyd.field_validator("outliers")
    def validate_outliers(cls, v: pd.Series) -> pd.Series:
        if v is not None:
            return ts_schema.validate(v)
        return v

    def concatenate(self, other: Timeseries) -> Timeseries:
        """Concatenate two Timeseries objects if they are considered equal."""
        if not isinstance(other, Timeseries):
            return NotImplemented

        if self == other:
            combined_ts = pd.concat([self.ts, other.ts]).sort_index()
            combined_ts = combined_ts[~combined_ts.index.duplicated(keep="first")]

            return self.model_copy(update={"ts": combined_ts})
        else:
            raise TimeseriesUnequal()

    def resample(
        self, freq: str, agg_func: Callable = pd.Series.mean, **resample_kwargs: Any
    ) -> Timeseries:
        """Resample the timeseries to a new frequency with a specified
        aggregation function.

        Parameters:
            freq (str): The new frequency for resampling the timeseries
                (e.g., 'D' for daily, 'W' for weekly).
            agg_func (Callable, optional): The aggregation function to apply
                after resampling. Defaults to pd.Series.mean.
            **resample_kwargs: Additional keyword arguments passed to the
                pandas.Series.resample method.

        Returns:
            Updated deep copy of the Timeseries object with the
                resampled timeseries data.
        """
        resampled_ts = self.ts.resample(freq, **resample_kwargs).apply(agg_func)

        return self.model_copy(update={"ts": resampled_ts}, deep=True)

    def transform(
        self,
        method: Literal[
            "difference",
            "log",
            "square_root",
            "box_cox",
            "standard_scaler",
            "minmax_scaler",
            "robust_scaler",
            "maxabs_scaler",
        ],
        **transformer_kwargs: Any,
    ) -> Timeseries:
        """Transforms the timeseries using the specified method.

        Parameters:
            method (str): The method to use for transformation ('minmax',
                'standard', 'robust').
            transformer_kwargs: Additional keyword arguments passed to the
                transformer definition. See gensor.preprocessing.

        Returns:
            Updated deep copy of the Timeseries object with the
                transformed timeseries data.
        """

        data, transformation = Transform(
            self.ts, method, **transformer_kwargs
        ).get_transformation()

        return self.model_copy(
            update={"ts": data, "transformation": transformation}, deep=True
        )

    def detect_outliers(
        self,
        method: Literal["iqr", "zscore", "isolation_forest", "lof"],
        rolling: bool = False,
        remove: bool = True,
        **kwargs: Any,
    ) -> Timeseries:
        """Detects outliers in the timeseries using the specified method.

        Parameters:
            method (Literal['iqr', 'zscore', 'isolation_forest', 'lof']): The
                method to use for outlier detection.
            **kwargs: Additional kewword arguments for OutlierDetection.

        Returns:
            Updated deep copy of the Timeseries object with outliers,
            optionally removed from the original timeseries.
        """
        self.outliers = OutlierDetection(self.ts, method, rolling, **kwargs).outliers

        if remove:
            filtered_ts = self.ts.drop(self.outliers.index)
            return self.model_copy(update={"ts": filtered_ts}, deep=True)

        else:
            return self

    def to_sql(self, db: DatabaseConnection) -> str:
        """Converts the timeseries to a list of dictionaries and uploads it to the database.

        The Timeseries data is uploaded to the SQL database by using the pandas
        `to_sql` method. Additionally, metadata about the timeseries is stored in the
        'timeseries_metadata' table.

        Args:
            db (DatabaseConnection): The database connection object.

        Returns:
            str: A message indicating the number of rows inserted into the database.
        """
        # Format the start timestamp as 'YYYYMMDDHHMMSS'
        timestamp_start_fmt = self.start.strftime("%Y%m%d%H%M%S")

        # Construct the schema name using the location, sensor, variable, unit, and timestamp
        schema_name = f"{self.location}_{self.sensor}_{self.variable}_{self.unit}_{timestamp_start_fmt}".lower()

        # Ensure the index is a pandas DatetimeIndex
        if isinstance(self.ts.index, pd.DatetimeIndex):
            utc_index = (
                self.ts.index.tz_convert("UTC")
                if self.ts.index.tz is not None
                else self.ts.index
            )
        else:
            message = "The index is not a DatetimeIndex and cannot be converted to UTC."
            raise TypeError(message)

        # Prepare the timeseries data as records for insertion
        series_as_records = list(
            zip(utc_index.strftime("%Y-%m-%dT%H:%M:%S%z"), self.ts, strict=False)
        )

        with db as con:
            # Create the timeseries table if it doesn't exist
            schema = db.create_table(schema_name, self.variable)

            # Ensure that the timeseries_metadata table exists
            metadata_schema = db.metadata.tables["__timeseries_metadata__"]

            if isinstance(schema, Table):
                # Insert the timeseries data
                stmt = sqlite_insert(schema).values(series_as_records)
                stmt = stmt.on_conflict_do_nothing(index_elements=["timestamp"])
                con.execute(stmt)
                con.commit()

                metadata_stmt = sqlite_insert(metadata_schema).values(
                    table_name=schema_name,
                    location=self.location,
                    sensor=self.sensor,
                    variable=self.variable,
                    unit=self.unit,
                    logger_alt=self.sensor_alt,
                    location_alt=self.sensor_alt,
                    timestamp_start=timestamp_start_fmt,
                    timestamp_end=self.end.strftime("%Y%m%d%H%M%S"),
                )

                metadata_stmt = metadata_stmt.on_conflict_do_update(
                    index_elements=["table_name"],
                    set_={
                        "timestamp_start": timestamp_start_fmt,
                        "timestamp_end": self.end.strftime("%Y%m%d%H%M%S"),
                    },
                )

                con.execute(metadata_stmt)
                con.commit()

        return f"{schema_name} table and metadata updated."

    def plot(
        self, include_outliers: bool = False, ax: Any = None, **plot_kwargs: Any
    ) -> tuple:
        """Plots the timeseries data.

        Args:
            include_outliers (bool): Whether to include outliers in the plot.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
                If None, a new figure and axes are created.
            **plot_kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            (fig, ax): Matplotlib figure and axes to allow further customization.
        """
        # Create new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        ax.plot(
            self.ts.index,
            self.ts,
            label=f"{self.location} ({self.sensor})",
            **plot_kwargs,
        )

        if include_outliers and self.outliers is not None:
            ax.scatter(
                self.outliers.index, self.outliers, color="red", label="Outliers"
            )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{self.variable} ({self.unit})")
        ax.set_title(f"{self.variable.capitalize()} at {self.location}")

        ax.legend()

        return fig, ax


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
            station (Optional[str]): The location of the station.
            sensor (Optional[str]): The sensor identifier.
            variable (Optional[str]): The variable being measured.

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

    # def align(self,
    #             freq: str = 'h',
    #             inplace: bool = True):
    #     """Aligns the timeseries to a common time axis.

    #     Args:
    #         freq (str): The target frequency for resampling.
    #         inplace (bool): Whether to update the timeseries in place. Defaults to True.
    #     """

    #     index_sets = [set(serie._resample(freq).index)
    #                     for serie in self.timeseries]

    #     # Find the intersection of all index sets to get the common dates
    #     common_dates = set.intersection(*index_sets)

    #     # Sort the common dates since set intersection will not preserve order
    #     common_dates = sorted(list(common_dates))

    #     aligned_series = []

    #     for serie in self.timeseries:
    #         serie.copy(deep=True)
    #         serie.timeseries = serie.timeseries.reindex(
    #             common_dates).dropna()

    #         aligned_series.append(serie)

    #     if inplace:
    #         self.timeseries = aligned_series
    #         return None
    #     else:
    #         aligned_series = Dataset(aligned_series)

    #     return aligned_series
