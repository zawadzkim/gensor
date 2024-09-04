"""Logic parsing CSV files from van Essen Instruments Divers."""

import re
from io import StringIO
from pathlib import Path
from typing import Any

import chardet
import pytz
from pandas import DataFrame, read_csv, to_datetime

from ..dtypes import Timeseries


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


def handle_timestamps(df: DataFrame, tz: str) -> DataFrame:
    """Converts the timestamps in the dataframe to the specified timezone.

    The timezone is obtained from the file metadata. If the timezone is UTC, the offset is extracted
    and the timestamps are converted to the corresponding timezone. If the timezone is not UTC, the
    timestamps are converted to UTC and then to the specified timezone.

    Args:
        df (pd.DataFrame): The dataframe with the data.
        tz (str): The timezone string obtained from the file metadata.
    """

    if tz.startswith("UTC"):
        offset_hours = int(tz[3:])
        timezone = pytz.FixedOffset(offset_hours * 60)
    else:
        timezone = pytz.UTC

    df.index = to_datetime(df.index).tz_localize("UTC").tz_convert(timezone)

    return df


def parse_vanessen_csv(path: Path, **kwargs) -> list[Any]:
    """Parses a van Essen csv file and returns a list of Timeseries objects. At this point it
    does not matter whether the file is a barometric or piezometric logger file.

    The function will use regex patterns to extract the serial number and station from the file. It is
    important to use the appropriate regex patterns, particularily for the station. If the default patterns
    are not working (whihc most likely will be the case), the user should provide their own patterns. The patterns
    can be provided as keyword arguments to the function and it is possible to use OR (|) in the regex pattern.

    Args:
        path (Path): The path to the file.
        **kwargs (dict): Optional keyword arguments to specify the regex patterns for the serial number and station.
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
            data = {k: re.search(v, text).group() for k, v in data.items()}
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

        timezone_match = re.search(
            kwargs.get("timezone_pattern", r"UTC[+-]?\d+"), text
        ).group()

        df = handle_timestamps(df, timezone_match)

        ts_list = [
            Timeseries(
                ts=df[col],
                variable=col,
                location=data.get("location"),
                sensor=data.get("sensor"),
                unit="cmH2O" if col == "pressure" else "degC",
            )
            for col in df.columns
        ]

    return ts_list
