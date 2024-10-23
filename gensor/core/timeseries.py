from __future__ import annotations

from typing import Any

import pandas as pd
import pandera as pa
import pydantic as pyd
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gensor.core.base import BaseTimeseries
from gensor.db import DatabaseConnection

ts_schema = pa.SeriesSchema(
    float,
    index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), coerce=False),
    coerce=True,
)


class Timeseries(BaseTimeseries):
    """Timeseries for groundwater sensor data

    Attributes:
        ts (pd.Series): The timeseries data.
        variable (Literal['temperature', 'pressure', 'conductivity', 'flux']):
            The type of the measurement.
        unit (Literal['degC', 'mmH2O', 'mS/cm', 'm/s']): The unit of
            the measurement.
        sensor (SensorInfo): The serial number of the sensor.

    Methods:
        validate_ts: if the pd.Series is not exactly what is required, coerce.
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
        fig, ax = super().plot(include_outliers=include_outliers, ax=ax, **plot_kwargs)

        ax.set_title(f"{self.variable.capitalize()} at {self.location} ({self.sensor})")

        return fig, ax
