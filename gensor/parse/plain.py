from pathlib import Path
from typing import Any

from pandas import read_csv

from ..dtypes import VARIABLE_TYPES_AND_UNITS, Timeseries
from .utils import detect_encoding, handle_timestamps


def parse_plain(path: Path, **kwargs: Any) -> list[Timeseries]:
    """Parse a simple csv without metadata header, just columns with variables

    Parameters:
        path (Path): The path to the file.

    Returns:
        list: A list of Timeseries objects.
    """

    column_names = kwargs.get("col_names", ["timestamp", "pressure", "temperature"])

    encoding = detect_encoding(path, num_bytes=10_000)

    df = read_csv(
        path,
        encoding=encoding,
        skipfooter=1,
        skip_blank_lines=True,
        header=None,
        skiprows=1,
        index_col="timestamp",
        names=column_names,
        engine="python",
    )

    df = handle_timestamps(df, kwargs.get("timezone", "UTC"))

    ts_list = []

    for col in df.columns:
        if col in VARIABLE_TYPES_AND_UNITS:
            unit = VARIABLE_TYPES_AND_UNITS[col][0]
            ts_list.append(
                Timeseries(
                    ts=df[col],
                    # Validation will be done in Pydantic
                    variable=col,  # type: ignore[arg-type]
                    location=kwargs["location"],
                    sensor=kwargs["sensor"],
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
