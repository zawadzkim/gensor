import re
from io import StringIO
from pathlib import Path

import chardet
from dateutil import tz
from pandas import DataFrame, read_csv, to_datetime


def _sniff_delimiter(header_line: str) -> str:
    """Guess the column delimiter from the data header line.

    Diver-Office exports are usually comma-separated, but some (e.g. certain
    barometric exports) use a semicolon or tab. Falls back to ',' .
    """
    counts = {sep: header_line.count(sep) for sep in (",", ";", "\t")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] else ","


def get_data(
    text: str, data_start: str, data_end: str, column_names: list
) -> DataFrame:
    """Search for data in the file.

    Parameters:
        text (str): string obtained from the CSV file.
        data_start (str): string marking the data header row.
        data_end (str): string marking the end of the data block. When it is not
            present (some exports omit the trailing marker), the data is read to
            the end of the file.
        column_names (list): list of expected column names.

    Returns:
        pd.DataFrame
    """

    start = text.find(data_start)
    if start == -1:
        message = f"Could not find the data header {data_start!r} in the file."
        raise ValueError(message)

    end = text.find(data_end, start)
    if end == -1:  # exports without the trailing marker: read to end of file
        end = len(text)

    block = text[start:end]
    sep = _sniff_delimiter(block.splitlines()[0])

    df = read_csv(
        StringIO(block),
        skiprows=1,
        header=None,
        names=column_names,
        index_col="timestamp",
        sep=sep,
    )

    return df


def get_metadata(text: str, patterns: dict) -> dict:
    """Search for metadata in the file header with given regex patterns.

    Parameters:
        text (str): string obtained from the CSV file.
        patterns (dict): regex patterns matching the location and sensor information.

    Returns:
        dict: metadata of the timeseries.
    """
    metadata = {}

    for k, v in patterns.items():
        match = re.search(v, text)
        metadata[k] = match.group() if match else None

    return metadata


def detect_encoding(path: Path, num_bytes: int = 1024) -> str:
    """Detect the encoding of a file using chardet.

    Parameters:
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

    Parameters:
        df (pd.DataFrame): The dataframe with timestamps.
        tz_string (str): A timezone string like 'UTC+1' or 'UTC-5'.

    Returns:
        pd.DataFrame: The dataframe with timestamps converted to UTC.
    """
    timezone = tz.gettz(tz_string)

    df.index = to_datetime(df.index).tz_localize(timezone)
    df.index = df.index.tz_convert("UTC")

    return df
