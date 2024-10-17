"""Fetching the data from various sources.

TODO: Fix up the read_from_sql() function to actually work properly.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy import select

from ..core.dataset import Dataset
from ..core.timeseries import Timeseries
from ..db.connection import DatabaseConnection
from ..exceptions import NoFilesToLoad
from ..parse import parse_plain, parse_vanessen_csv


def read_from_csv(
    path: Path, file_format: Literal["vanessen", "plain"] = "vanessen", **kwargs: Any
) -> Dataset:
    """Loads the data from csv files with given file_format and returns a list of Timeseries objects.

    Parameters:
        path (Path): The path to the file or directory containing the files.
        **kwargs (dict): Optional keyword arguments passed to the parsers:
            * serial_number_pattern (str): The regex pattern to extract the serial number from the file.
            * location_pattern (str): The regex pattern to extract the station from the file.
            * col_names (list): The column names for the dataframe.
            * location (str): Name of the location of the timeseries.
            * sensor (str): Sensor serial number.
    """

    parsers = {
        "vanessen": parse_vanessen_csv,
        "plain": parse_plain,
        # more parser to be implemented
    }

    if not isinstance(path, Path):
        message = "The path argument must be a Path object."
        raise TypeError(message)

    if path.is_dir() and not any(
        file.is_file() and file.suffix.lower() == ".csv" for file in path.iterdir()
    ):
        raise NoFilesToLoad()

    files = (
        [file for file in path.iterdir() if file.is_file() and file.suffix == ".csv"]
        if path.is_dir()
        else [path]
        if path.suffix == ".csv"
        else []
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
    timestamp_start: pd.Timestamp | None = None,
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

    def _read_from_sql(schema_name: str) -> Timeseries:
        with db as con:
            schema = db.metadata.tables[schema_name]
            metadata_table = db.metadata.tables["__timeseries_metadata__"]
            data = select(schema)

            ts = pd.read_sql(
                data,
                con=con,
                parse_dates={"timestamp": "%Y-%m-%dT%H:%M:%S%z"},
                index_col="timestamp",
            ).squeeze()

        if ts.empty:
            message = f"No data found in table {schema_name}"
            raise ValueError(message)

        # Retrieve metadata associated with this timeseries
        metadata_query = select(metadata_table).where(
            metadata_table.c.table_name == schema_name
        )
        metadata_result = con.execute(metadata_query).fetchone()

        if not metadata_result:
            message = f"No metadata found for table {schema_name}"
            raise ValueError(message)

        location = metadata_result[2]
        sensor = metadata_result[3]
        variable = metadata_result[4]
        unit = metadata_result[5]
        sensor_alt = metadata_result[6]
        # location_alt = metadata_result[7]

        ts_object = Timeseries(
            ts=ts,
            variable=variable,
            location=location,
            sensor=sensor,
            unit=unit,
            sensor_alt=sensor_alt,
        )

        return ts_object

    # fmt: off
    if load_all:
        schemas = db.get_tables()
        if schemas:
            timeseries = [_read_from_sql(ts_name)
                          for ts_name in schemas]

            return Dataset(timeseries=[ts for ts in timeseries if ts is not None])
        else:
            return Dataset()
    else:
        if isinstance(timestamp_start, pd.Timestamp):
            timestamp_start_fmt = timestamp_start.strftime("%Y%m%d%H%M%S")
        schema_name = (
            f"{location}_{sensor}_{variable}_{unit}_{timestamp_start_fmt}".lower()
        )
        return _read_from_sql(schema_name)


# fmt: on


def read_from_api() -> Dataset:
    """Fetch data from the API."""
    return NotImplemented
