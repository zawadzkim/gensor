import re
from io import StringIO
from pathlib import Path

import chardet
from dateutil import tz
from pandas import DataFrame, read_csv, to_datetime


def get_data(
    text: str, data_start: str, data_end: str, column_names: list
) -> DataFrame:
    data_io = StringIO(text[text.index(data_start) : text.index(data_end)])

    df = read_csv(
        data_io, skiprows=1, header=None, names=column_names, index_col="timestamp"
    )

    return df


def get_metadata(text: str, patterns: dict) -> dict:
    """Search for metadata in the file header with given regex patterns."""
    metadata = {}

    for k, v in patterns.items():
        match = re.search(v, text)
        metadata[k] = match.group() if match else None

    if metadata["sensor"] is None or metadata["location"] is None:
        return {}

    return metadata


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
