"""Fetching the data from various sources.

TODO: Fix up the read_from_sql() function to actually work properly.
"""

import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy import select

from ..core.dataset import Dataset
from ..core.timeseries import Timeseries
from ..db.connection import DatabaseConnection
from ..parse import parse_plain, parse_vanessen_csv

logger = logging.getLogger(__name__)


def read_from_csv(
    path: Path, file_format: Literal["vanessen", "plain"] = "vanessen", **kwargs: Any
) -> Dataset | Timeseries:
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
        logger.info("No CSV files found. Operation skipped.")
        return Dataset()

    files = (
        [
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() == ".csv"
        ]
        if path.is_dir()
        else [path]
        if path.suffix.lower() == ".csv"
        else []
    )

    if not files:
        logger.info("No CSV files found. Operation skipped.")
        return Dataset()

    parser = parsers[file_format]

    ds: Dataset = Dataset()

    for f in files:
        logger.info(f"Loading file: {f}")
        ts_in_file = parser(f, **kwargs)
        ds.add(ts_in_file)

    # If there is only one Timeseries in Dataset (as in the condition), ds[0] will always
    # be a Timeseries; so the line below does not introduce potential None in the return
    return ds[0] if len(ds) == 1 else ds  # type: ignore[return-value]


def read_from_sql(
    db: DatabaseConnection,
    load_all: bool,
    location: str | None = None,
    variable: str | None = None,
    unit: str | None = None,
    timestamp_start: pd.Timestamp | None = None,
    timestamp_stop: pd.Timestamp | None = None,
    **kwargs: dict,
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

    def _read_from_sql(schema_name: str) -> Any:
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

        # Core metadata extraction
        core_metadata = {
            "location": metadata_result[2],
            "variable": metadata_result[3],
            "unit": metadata_result[4],
        }

        extra_metadata = metadata_result[7] or {}
        cls = metadata_result[8]

        metadata = {**core_metadata, **extra_metadata}

        module_name, class_name = cls.rsplit(".", 1)
        module = import_module(module_name)

        TimeseriesClass = getattr(module, class_name)
        ts_object = TimeseriesClass(ts=ts, **metadata)

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

        schema_name = (
            f"{location}_{variable}_{unit}".lower()
        )
        return _read_from_sql(schema_name)


# fmt: on


def read_from_api() -> Dataset:
    """Fetch data from the API."""
    return NotImplemented
