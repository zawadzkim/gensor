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
    load_all: bool = True,
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
        variable (str): The measurement type.
        unit (str): The unit of the measurement.
        timestamp_start (pd.Timestamp, optional): Start timestamp filter.
        timestamp_stop (pd.Timestamp, optional): End timestamp filter.
        **kwargs (dict): Any additional filters matching attributes of the particular
            timeseries.

    Returns:
        Dataset: Dataset with retrieved objects or an empty Dataset.
    """

    def _read_data_from_schema(schema_name: str) -> Any:
        """Read data from the table and apply the timestamp filter.

        Parameters:
            schema_name (str): name of the schema in SQLite database.

        Returns:
            pd.Series: results of the query or an empty pd.Series if none are found.
        """
        with db as con:
            schema = db.metadata.tables[schema_name]
            data_query = select(schema)

            if timestamp_start or timestamp_stop:
                if timestamp_start:
                    data_query = data_query.where(schema.c.timestamp >= timestamp_start)
                if timestamp_stop:
                    data_query = data_query.where(schema.c.timestamp <= timestamp_stop)

            ts = pd.read_sql(
                data_query,
                con=con,
                parse_dates={"timestamp": "%Y-%m-%dT%H:%M:%S%z"},
                index_col="timestamp",
            ).squeeze()

        if ts.empty:
            message = f"No data found in table {schema_name}"
            logger.warning(message)

        return ts

    def _create_object(data: pd.Series, metadata: dict) -> Any:
        """Create the appropriate object for timeseries."""

        core_metadata = {
            "location": metadata["location"],
            "variable": metadata["variable"],
            "unit": metadata["unit"],
        }

        extra_metadata = metadata.get("extra", {})

        ts_metadata = {**core_metadata, **extra_metadata}

        cls = metadata["cls"]
        module_name, class_name = cls.rsplit(".", 1)
        module = import_module(module_name)

        TimeseriesClass = getattr(module, class_name)
        ts_object = TimeseriesClass(ts=data, **ts_metadata)

        return ts_object

    metadata_df = (
        db.get_timeseries_metadata(
            location=location, variable=variable, unit=unit, **kwargs
        )
        if not load_all
        else db.get_timeseries_metadata()
    )

    if metadata_df.empty:
        message = "No schemas matched the specified filters."
        raise ValueError(message)

    timeseries_list = []

    for row in metadata_df.to_dict(orient="records"):
        try:
            schema_name = row.pop("table_name")
            data = _read_data_from_schema(schema_name)
            timeseries_obj = _create_object(data, row)
            timeseries_list.append(timeseries_obj)
        except (ValueError, TypeError):
            logger.exception(f"Skipping schema {schema_name} due to error.")

    return Dataset(timeseries=timeseries_list) if timeseries_list else Dataset()


def read_from_api() -> Dataset:
    """Fetch data from the API."""
    return NotImplemented
