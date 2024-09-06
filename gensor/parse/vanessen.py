"""Logic parsing CSV files from van Essen Instruments Divers."""

import re
from io import StringIO
from pathlib import Path
from typing import Any

import chardet
from dateutil import tz
from pandas import DataFrame, read_csv, to_datetime

from ..dtypes import VARIABLE_TYPES_AND_UNITS, Timeseries


def detect_encoding(path: Path, num_bytes: int = 1024) -> str:
    """Detect the encoding of a file using chardet.

    Args:
        path (Path): The path to the file.
        num_bytes (int): Number of bytes to read for encoding detection (default is 1024).

    Returns:
        str: The detected encoding of the file.
    """
    with path.open("rb") as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result["encoding"] or "utf-8"


def handle_timestamps(df: DataFrame, tz_string: str) -> DataFrame:
    """Converts timestamps in the dataframe to the specified timezone (e.g., 'UTC+1').

    Args:
        df (pd.DataFrame): The dataframe with timestamps.
        tz_string (str): A timezone string like 'UTC+1' or 'UTC-5'.

    Returns:
        pd.DataFrame: The dataframe with timestamps converted to UTC.
    """
    timezone = tz.gettz(tz_string)

    df.index = to_datetime(df.index).tz_localize(timezone)
    df.index = df.index.tz_convert("UTC")

    return df


def parse_vanessen_csv(path: Path, **kwargs: Any) -> list[Timeseries]:
    """Parses a van Essen csv file and returns a list of Timeseries objects. At this point it
    does not matter whether the file is a barometric or piezometric logger file.

    The function will use regex patterns to extract the serial number and station from the file. It is
    important to use the appropriate regex patterns, particularily for the station. If the default patterns
    are not working (whihc most likely will be the case), the user should provide their own patterns. The patterns
    can be provided as keyword arguments to the function and it is possible to use OR (|) in the regex pattern.

    !!! warning

        A better check for the variable type and units has to be implemented.

    Parameters:
        path (Path): The path to the file.

    Other Parameters:
        serial_number_pattern (str): The regex pattern to extract the serial number from the file.
        location_pattern (str): The regex pattern to extract the station from the file.
        col_names (list): The column names for the dataframe.

    Returns:
        list: A list of Timeseries objects.
    """

    data = {
        "sensor": kwargs.get("serial_number_pattern", r"[A-Za-z]{2}\d{3,4}"),
        "location": kwargs.get(
            "location_pattern", r"[A-Za-z]{2}\d{2}[A-Za-z]{1}|Barodiver"
        ),
    }

    column_names = kwargs.get("col_names", ["timestamp", "pressure", "temperature"])

    encoding = detect_encoding(path, num_bytes=10_000)

    with path.open(mode="r", encoding=encoding) as f:
        text = f.read()

        try:
            data = {
                k: (match.group() if (match := re.search(v, text)) else None)
                for k, v in data.items()
            }

        except AttributeError:
            print(
                f"Skipping file {path} due to missing patterns. If this is not expected, please provide the correct patterns."
            )
            return []

        data_io = StringIO(
            text[
                text.index("Date/time") : text.index(
                    "END OF DATA FILE OF DATALOGGER FOR WINDOWS"
                )
            ]
        )

        df = read_csv(
            data_io, skiprows=1, header=None, names=column_names, index_col="timestamp"
        )
        timezone_pattern = kwargs.get("timezone_pattern", r"UTC[+-]?\d+")
        timezone_match = re.search(timezone_pattern, text)

        timezone = timezone_match.group() if timezone_match else "UTC"

        df = handle_timestamps(df, timezone)

        ts_list = []

        for col in df.columns:
            if col in VARIABLE_TYPES_AND_UNITS:
                unit = VARIABLE_TYPES_AND_UNITS[col][0]
                ts_list.append(
                    Timeseries(
                        ts=df[col],
                        # Validation will be done in Pydantic
                        variable=col,  # type: ignore[arg-type]
                        location=data.get("location"),
                        sensor=data.get("sensor"),
                        # Validation will be done in Pydantic
                        unit=unit,  # type: ignore[arg-type]
                    )
                )
            else:
                message = (
                    "Unsupported variable: {col}. Please provide a valid variable type."
                )
                raise ValueError(message)

    return ts_list
