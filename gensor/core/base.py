from __future__ import annotations

from typing import Any, Literal, TypeVar

import pandas as pd
import pandera as pa
import pydantic as pyd
from matplotlib import pyplot as plt
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gensor.analysis.outliers import OutlierDetection
from gensor.core.indexer import TimeseriesIndexer
from gensor.db import DatabaseConnection
from gensor.exceptions import TimeseriesUnequal
from gensor.processing.transform import Transformation

T = TypeVar("T", bound="BaseTimeseries")

ts_schema = pa.SeriesSchema(
    float,
    index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), coerce=False),
    coerce=True,
)


class BaseTimeseries(pyd.BaseModel):
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

    def __eq__(self, other: T) -> bool:
        """Check equality based on location, sensor, variable, unit and sensor_alt."""
        if not isinstance(other, T):
            return NotImplemented

        return (
            self.variable == other.variable
            and self.unit == other.unit
            and self.location == other.location
        )

    def __getattr__(self, attr: Any) -> Any:
        """Delegate attribute access to the underlying pandas Series if it exists.

        Special handling is implemented for pandas indexer.
        """
        if attr == "loc":
            return TimeseriesIndexer(self, self.ts.loc)

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
        validated_ts = ts_schema.validate(v)

        return validated_ts

    @pyd.field_validator("outliers")
    def validate_outliers(cls, v: pd.Series) -> pd.Series:
        if v is not None:
            return ts_schema.validate(v)
        return v

    def concatenate(self, other: T) -> T:
        """Concatenate two Timeseries objects if they are considered equal."""
        if not isinstance(other, T):
            return NotImplemented

        if self == other:
            combined_ts = pd.concat([self.ts, other.ts]).sort_index()
            combined_ts = combined_ts[~combined_ts.index.duplicated(keep="first")]

            return self.model_copy(update={"ts": combined_ts})
        else:
            raise TimeseriesUnequal()

    def resample(
        self,
        freq: Any,
        agg_func: Any = pd.Series.mean,
        **resample_kwargs: Any,
    ) -> T:
        """Resample the timeseries to a new frequency with a specified
        aggregation function.

        Parameters:
            freq (Any): The offset string or object representing target conversion
                (e.g., 'D' for daily, 'W' for weekly).
            agg_func (Any): The aggregation function to apply
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
    ) -> T:
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

        data, transformation = Transformation(
            self.ts, method, **transformer_kwargs
        ).get_transformation()

        return self.model_copy(
            update={"ts": data, "transformation": transformation}, deep=True
        )

    def detect_outliers(
        self,
        method: Literal["iqr", "zscore", "isolation_forest", "lof"],
        rolling: bool = False,
        window: int = 6,
        remove: bool = True,
        **kwargs: Any,
    ) -> T:
        """Detects outliers in the timeseries using the specified method.

        Parameters:
            method (Literal['iqr', 'zscore', 'isolation_forest', 'lof']): The
                method to use for outlier detection.
            **kwargs: Additional kewword arguments for OutlierDetection.

        Returns:
            Updated deep copy of the Timeseries object with outliers,
            optionally removed from the original timeseries.
        """
        self.outliers = OutlierDetection(
            self.ts, method, rolling, window, **kwargs
        ).outliers

        if remove:
            filtered_ts = self.ts.drop(self.outliers.index)
            return self.model_copy(update={"ts": filtered_ts}, deep=True)

        else:
            return self

    def mask_with(
        self, other: T | pd.Series, mode: Literal["keep", "remove"] = "remove"
    ) -> T:
        """
        Removes records not present in 'other' by index.

        Parameters:
            other (Timeseries): Another Timeseries whose indices are used to mask the current one.
            mode (Literal['keep', 'remove']):
                - 'keep': Retains only the overlapping data.
                - 'remove': Removes the overlapping data.

        Returns:
            Timeseries: A new Timeseries object with the filtered data.
        """
        if isinstance(other, pd.Series):
            mask = other
        elif isinstance(other, T):
            mask = other.ts

        if mode == "keep":
            masked_data = self.ts[self.ts.index.isin(mask.index)]
        elif mode == "remove":
            masked_data = self.ts[~self.ts.index.isin(mask.index)]
        else:
            message = f"Invalid mode: {mode}. Use 'keep' or 'remove'."
            raise ValueError(message)

        return self.model_copy(update={"ts": masked_data}, deep=True)

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
        schema_name = (
            f"{self.location}_{self.variable}_{self.unit}_{timestamp_start_fmt}".lower()
        )

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
                    variable=self.variable,
                    unit=self.unit,
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
            label=f"{self.location}",
            **plot_kwargs,
        )

        if include_outliers and self.outliers is not None:
            ax.scatter(
                self.outliers.index, self.outliers, color="red", label="Outliers"
            )
        plt.xticks(rotation=45)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{self.variable} ({self.unit})")
        ax.set_title(f"{self.variable.capitalize()} at {self.location}")

        ax.legend()

        return fig, ax
