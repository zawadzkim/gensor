"""Fetching the data from various sources."""

from pathlib import Path
from typing import Literal

from pandas import read_sql
from sqlalchemy import MetaData, Table, select

from .db.connection import DatabaseConnection
from .dtypes import Dataset, Timeseries
from .exceptions import NoFilesToLoad
from .parse import parse_vanessen_csv


def read_from_csv(path: Path, file_format: Literal["vanessen"] = "vanessen", **kwargs):
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
    all_ts = []
    for f in files:
        print(f"Loading file: {f}")
        all_ts.extend(parser(f, **kwargs))

    if len(all_ts) > 1:
        return Dataset(timeseries=all_ts)
    else:
        return all_ts[0]


def read_from_sql(
    db: DatabaseConnection, location: str, sensor: str, variable: str, unit: str
) -> list[Timeseries]:
    """Returns the timeseries from a sql database.

    Parameters:
        db (DatabaseConnection): The database connection object
        location (str): The station name
        sensor (str): Sensor name
        variable (str): The measurement type
        unit (str): Unit of the measurement

    """
    metadata = MetaData()
    schema = Table(f"{location}_{sensor}_{variable}", metadata)

    query = select(schema)
    df = read_sql(query, con=db.engine)

    ts_object = Timeseries(
        timeseries=df, variable=variable, location=location, sensor=sensor, unit=unit
    )

    return ts_object


def read_from_api() -> Dataset:
    """Fetch data from the API."""
    return NotImplemented
