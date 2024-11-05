"""Test data for Gensor package:

Attributes:

    all (Path): The whole directory of test groundwater sensor data.
    baro (Path): Timeseries of barometric pressure measurements.
    pb01a (Path): Timeseries of a submerged logger.
    pb02a_plain (Path): Timeseries from PB02A with the metadata removed.

"""

from importlib import resources
from importlib.abc import Traversable

all_paths: Traversable = resources.files(__name__)
"""The whole directory of test groundwater sensor data."""

baro: Traversable = all_paths / "Barodiver_220427183008_BY222.csv"
"""Timeseries of barometric pressure measurements."""

pb01a: Traversable = all_paths / "PB01A_moni_AV319_220427183019_AV319.csv"
"""Timeseries of a submerged logger."""

pb02a_plain: Traversable = all_paths / "PB02A_plain.csv"
"""Timeseries from PB02A with the metadata removed."""
