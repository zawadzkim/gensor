"""Fetching the data from various sources.

TODO: Fix up the read_from_sql() function to actually work properly.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy import select

from .db.connection import DatabaseConnection
from .dtypes import Dataset, Timeseries
from .exceptions import NoFilesToLoad
from .parse import parse_vanessen_csv


def read_from_csv(
    path: Path, file_format: Literal["vanessen"] = "vanessen", **kwargs: Any
) -> Dataset:
    """Loads the data from the Van Essen CSV file(s) and returns a list of Timeseries objects.

    Args:
        path (Path): The path to the file or directory containing the files.
        **kwargs (dict): Optional keyword arguments passed to `parse_vanessen_csv()` to specify the regex patterns for the serial number and station.
            serial_number_pattern (str): The regex pattern to extract the serial number from the file.
            location_pattern (str): The regex pattern to extract the station from the file.
            col_names (list): The column names for the dataframe.
    """

    parsers = {
        "vanessen": parse_vanessen_csv,
    }

    if not isinstance(path, Path):
        message = "The path argument must be a Path object."
        raise TypeError(message)

    if path.is_dir() and not any(path.iterdir()):
        raise NoFilesToLoad()

    files = (
        [file for file in path.iterdir() if file.is_file()] if path.is_dir() else [path]
    )

    parser = parsers[file_format]
    ds = Dataset()
    for f in files:
        print(f"Loading file: {f}")
        ts_in_file = parser(f, **kwargs)
        ds.add(ts_in_file)

    return ds


def read_from_sql(
    db: DatabaseConnection,
    load_all: bool,
    location: str | None = None,
    sensor: str | None = None,
    variable: str | None = None,
    unit: str | None = None,
) -> Timeseries | Dataset:
    """Returns the timeseries or a dataset from a SQL database.

    Parameters:
        db (DatabaseConnection): The database connection object.
        load_all (bool): Whether to load all timeseries from the database.
        location (str): The station name.
        sensor (str): The sensor name.
        variable (str): The measurement type.
        unit (str): The unit of the measurement.

    Returns:
        Timeseries: The Timeseries object retrieved from the database.

    Raises:
        ValueError: If the DataFrame cannot be retrieved or if it's empty.
        TypeError: If the retrieved data is not a DataFrame or is of incorrect type.
    """

    def _read_from_sql(
        location: str, sensor: str, variable: str, unit: str
    ) -> Timeseries:
        schema_name = f"{location}_{sensor}_{variable}_{unit}".lower()

        with db as con:
            schema = db.metadata.tables[schema_name]
            query = select(schema)
            ts = pd.read_sql(
                query,
                con=con,
                parse_dates={"timestamp": "%Y-%m-%dT%H:%M:%S%z"},
                index_col="timestamp",
            ).squeeze()
        if ts.empty:
            message = f"No data found in table {schema_name}"
            raise ValueError(message)

        # Variable and type validation are handled by pydantic model
        ts_object = Timeseries(
            ts=ts,
            variable=variable,  # type: ignore[arg-type]
            location=location,
            sensor=sensor,
            unit=unit,  # type: ignore[arg-type]
        )

        return ts_object

    # fmt: off
    if load_all:
        schemas = db.get_tables()
        if schemas:
            timeseries = [_read_from_sql(*ts_name.split("_"))
                          for ts_name in schemas]

            return Dataset(timeseries=[ts for ts in timeseries if ts is not None])
        else:
            return Dataset()
    else:

        return _read_from_sql(location, sensor, variable, unit)  # type: ignore[arg-type]


# fmt: on


def read_from_api() -> Dataset:
    """Fetch data from the API."""
    return NotImplemented
