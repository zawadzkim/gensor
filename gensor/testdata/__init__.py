"""Test data for Gensor package:

Attributes:

    all (Path): The whole directory of test groundwater sensor data.
    baro (Path): Timeseries of barometric pressure measurements.
    pb01a (Path): Timeseries of a submerged logger.
    pb02a_plain (Path): Timeseries from PB02A with the metadata removed.

"""

from importlib import resources
from pathlib import Path

all: Path = resources.files(__name__)
"""The whole directory of test groundwater sensor data."""

baro: Path = all / "Barodiver_220427183008_BY222.csv"
"""Timeseries of barometric pressure measurements."""

pb01a: Path = all / "PB01A_moni_AV319_220427183019_AV319.csv"
"""Timeseries of a submerged logger."""

pb02a_plain: Path = all / "PB02A_plain.csv"
"""Timeseries from PB02A with the metadata removed."""
