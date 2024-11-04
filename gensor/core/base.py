from __future__ import annotations

from typing import Any, Literal, TypeVar
import hashlib

import pandas as pd
import pandera as pa
import pydantic as pyd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    """Generic base class for timeseries with metadata.

    Timeseries is a series of measurements of a single variable, in the same unit, from a
    single location with unique timestamps.

    Attributes:
        ts (pd.Series): The timeseries data.
        variable (Literal['temperature', 'pressure', 'conductivity', 'flux']):
            The type of the measurement.
        unit (Literal['degC', 'mmH2O', 'mS/cm', 'm/s']): The unit of
            the measurement.
        outliers (pd.Series): Measurements marked as outliers.
        transformation (Any): Metadata of transformation the timeseries undergone.

    Methods:
        validate_ts: if the pd.Series is not exactly what is required, coerce.
    """

    model_config = pyd.ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    ts: pd.Series = pyd.Field(repr=False, exclude=True)
    variable: Literal[
        "temperature", "pressure", "conductivity", "flux", "head", "depth"
    ]
    unit: Literal["degc", "cmh2o", "ms/cm", "m/s", "m asl", "m"]
    location: str | None = None
    outliers: pd.Series | None = pyd.Field(default=None, repr=False, exclude=True)
    transformation: Any = pyd.Field(default=None, repr=False, exclude=True)

    @pyd.computed_field()  # type: ignore[prop-decorator]
    @property
    def start(self) -> pd.Timestamp | Any:
        return self.ts.index.min()

    @pyd.computed_field()  # type: ignore[prop-decorator]
    @property
    def end(self) -> pd.Timestamp | Any:
        return self.ts.index.max()

    @pyd.field_serializer("start", "end")
    def serialize_timestamps(self, value: pd.Timestamp | None) -> str | None:
        """Serialize `pd.Timestamp` to ISO format."""
        return value.strftime("%Y%m%d%H%M%S") if value is not None else None

    def __eq__(self, other: object) -> bool:
        """Check equality based on location, sensor, variable, unit and sensor_alt."""
        if not isinstance(other, BaseTimeseries):
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

        if attr == "iloc":
            return TimeseriesIndexer(self, self.ts.iloc)

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

    def concatenate(self: T, other: T) -> T:
        """Concatenate two Timeseries objects if they are considered equal."""
        if not isinstance(other, type(self)):
            return NotImplemented

        if self == other:
            combined_ts = pd.concat([self.ts, other.ts]).sort_index()
            combined_ts = combined_ts[~combined_ts.index.duplicated(keep="first")]

            return self.model_copy(update={"ts": combined_ts})
        else:
            raise TimeseriesUnequal()

    def resample(
        self: T,
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
        self: T,
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
        self: T,
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
        self: T, other: T | pd.Series, mode: Literal["keep", "remove"] = "remove"
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
        elif isinstance(other, BaseTimeseries):
            mask = other.ts

        if mode == "keep":
            masked_data = self.ts[self.ts.index.isin(mask.index)]
        elif mode == "remove":
            masked_data = self.ts[~self.ts.index.isin(mask.index)]
        else:
            message = f"Invalid mode: {mode}. Use 'keep' or 'remove'."
            raise ValueError(message)

        return self.model_copy(update={"ts": masked_data}, deep=True)

    def to_sql(self: T, db: DatabaseConnection) -> str:
        """Converts the timeseries to a list of dictionaries and uploads it to the database.

        The Timeseries data is uploaded to the SQL database by using the pandas
        `to_sql` method. Additionally, metadata about the timeseries is stored in the
        'timeseries_metadata' table.

        Parameters:
            db (DatabaseConnection): The database connection object.

        Returns:
            str: A message indicating the number of rows inserted into the database.
        """

        def separate_metadata() -> tuple:
            _core_metadata_fields = {"location", "variable", "unit", "start", "end"}

            core_metadata = self.model_dump(include=_core_metadata_fields)
            core_metadata.update({
                "cls": f"{self.__module__}.{self.__class__.__name__}"
            })

            extra_metadata = self.model_dump(exclude=_core_metadata_fields)

            return core_metadata, extra_metadata

        timestamp_start_fmt = self.start.strftime("%Y%m%d%H%M%S")
        timestamp_end_fmt = self.end.strftime("%Y%m%d%H%M%S")

        schema_name = f"{self.location}_{self.variable}_{self.unit}".lower()

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

        series_as_records = list(
            zip(utc_index.strftime("%Y-%m-%dT%H:%M:%S%z"), self.ts, strict=False)
        )

        # Extra metadata are attributes additional to BaseTimeseries
        core_metadata, extra_metadata = separate_metadata()

        metadata_entry = {
            **core_metadata,
            "extra": extra_metadata,
            "table_name": schema_name,
        }

        unique_hash = hashlib.sha1(schema_name.encode("utf-8")).hexdigest()[:5]
        
        created_table = db.get_timeseries_metadata(
            location=self.location,
            variable=self.variable,
            unit=self.unit,
            **extra_metadata
        )

        
        
        with db as con:
            schema = db.create_table(schema_name, self.variable)
            metadata_schema = db.metadata.tables["__timeseries_metadata__"]

            if isinstance(schema, Table):
                stmt = sqlite_insert(schema).values(series_as_records)
                stmt = stmt.on_conflict_do_nothing(index_elements=["timestamp"])
                con.execute(stmt)

                metadata_stmt = sqlite_insert(metadata_schema).values(metadata_entry)
                metadata_stmt = metadata_stmt.on_conflict_do_update(
                    index_elements=["table_name"],
                    set_={
                        "start": timestamp_start_fmt,
                        "end": timestamp_end_fmt,
                    },
                )
                con.execute(metadata_stmt)

            # Commit all changes at once
            con.commit()

        return f"{schema_name} table and metadata updated."

    def plot(
        self: T,
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

        plot_kwargs = plot_kwargs or {}
        legend_kwargs = legend_kwargs or {}

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            # mypy complained that the get_figure() can return None, but there is no
            # situation here in which this could be the case.
            fig = ax.get_figure()  # type: ignore [assignment]

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
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{self.variable} ({self.unit})")
        ax.set_title(f"{self.variable.capitalize()} at {self.location}")

        ax.legend(**legend_kwargs)

        return fig, ax
