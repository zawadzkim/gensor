"""Logic parsing CSV files from van Essen Instruments Divers."""

import logging
import re
from pathlib import Path
from typing import Any

from ..config import VARIABLE_TYPES_AND_UNITS
from ..core.timeseries import Timeseries
from .utils import detect_encoding, get_data, get_header_fields, handle_timestamps

logger = logging.getLogger(__name__)


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

    patterns = {
        "sensor": kwargs.get("serial_number_pattern", r"[A-Za-z]{1,2}\d{3,4}"),
        "location": kwargs.get(
            "location_pattern", r"[A-Za-z]{2}\d{2}[A-Za-z]{1}|Barodiver"
        ),
        "timezone": kwargs.get("timezone_pattern", r"UTC[+-]?\d+"),
    }

    column_names = kwargs.get("col_names", ["timestamp", "pressure", "temperature"])

    encoding = detect_encoding(path, num_bytes=10_000)

    with path.open(mode="r", encoding=encoding) as f:
        text = f.read()

        fields = get_header_fields(text)

        def pick(pattern: str, raw: str | None, override: str | None) -> str | None:
            """Resolve a metadata value from a labelled header field.

            An explicit ``location=``/``sensor=`` kwarg always wins. Otherwise the
            pattern is matched against *that field's value only* (so e.g. the serial
            ``AZ066`` is pulled out of ``..00-AZ066  219.`` and the location ``PB16A``
            out of ``pb16a_moni_az066``), falling back to the verbatim field value
            when the pattern does not match (e.g. ``FL1`` / ``barodiver`` locations).
            """
            if override:
                return override
            if not raw:
                return None
            match = re.search(pattern, raw)
            return match.group() if match else raw

        # Read the labelled header fields directly — far more reliable than matching a
        # regex against the whole file, which can grab a stray token from the embedded
        # FILENAME path (e.g. a folder name) instead of the real serial.
        location = pick(
            patterns["location"], fields.get("Location"), kwargs.get("location")
        )
        sensor = pick(
            patterns["sensor"], fields.get("Serial number"), kwargs.get("sensor")
        )
        tz_match = re.search(patterns["timezone"], text)
        timezone = tz_match.group() if tz_match else "UTC"

        if location is None or sensor is None:
            logger.info(
                f"Skipping file {path} due to missing metadata "
                "(pass location=/sensor= to override)."
            )
            return []

        data_start = "Date/time"
        data_end = "END OF DATA FILE"

        df = get_data(text, data_start, data_end, column_names)

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
                        location=location,
                        sensor=sensor,
                        # Validation will be done in Pydantic
                        unit=unit,  # type: ignore[arg-type]
                    )
                )
            else:
                message = f"Unsupported variable: {col}. Please provide a valid variable type."
                raise ValueError(message)

    return ts_list
