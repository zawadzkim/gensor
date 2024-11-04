"""
Fixtures for all tests.

Fixtures:
    db: Creates a temporary database file for each test and ensures cleanup after the test.
    synthetic_submerged_timeseries: Provides a synthetic Timeseries with pressure data to simulate submerged sensor readings.
    extended_synthetic_submerged_pressure_timeseries: Provides an extended synthetic pressure Timeseries to add new measurements for testing.
    barometric_value: Provides a static barometric pressure value for testing.
    synthetic_barometric_pressure_timeseries: Generates a synthetic barometric pressure Timeseries for testing compensation mechanisms.
    synthetic_expected_compensated_timeseries: Provides a pre-calculated expected compensated Timeseries for comparison in tests.
    synthetic_timeseries_with_none: Provides a Timeseries with some `None` values to test how the system handles missing data.
    synthetic_temperature_timeseries: Generates a synthetic Timeseries representing temperature data for testing non-pressure variables.
    baro_timeseries: Provides actual barometric pressure Timeseries data from a van Essen Diver, read from a CSV.
    pb01a_timeseries: Provides actual well Timeseries data from a van Essen Diver for location PB01A, read from a CSV.
    pb02a_plain_timeseries: Provides well Timeseries data from a van Essen Diver (PB02A) with metadata removed to simulate a plain file, read from a CSV.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from gensor import read_from_csv
from gensor.core.dataset import Dataset
from gensor.core.timeseries import Timeseries
from gensor.db import DatabaseConnection

# ====================================== Database ======================================


@pytest.fixture
def db(tmp_path):
    """Fixture to create a temporary database file for each test."""
    db_connection = DatabaseConnection(db_directory=tmp_path, db_name="test_db.sqlite")

    yield db_connection

    db_path = tmp_path / "test_db.sqlite"
    if db_path.exists():
        os.remove(db_path)


# ================================ Synthetic timeseries ================================


@pytest.fixture
def synthetic_submerged_timeseries():
    """A valid pressure Timeseries."""
    data = pd.Series(
        [1313.00, 1312.00, 1310.00],
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmh2o",
        location="Station A",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def extended_synthetic_submerged_pressure_timeseries():
    """New pressure measurements for the synthetic_submerged_timeseries."""
    data = pd.Series(
        [1314.25, 1315.5, 1316.0],
        index=pd.date_range("2024-01-02", periods=3, freq="h", tz="UTC"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmh2o",
        location="Station A",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def barometric_value():
    return 1010.0


@pytest.fixture
def synthetic_barometric_pressure_timeseries(barometric_value):
    """Synthetic barometric pressure Timeseries."""
    data = pd.Series(
        [barometric_value] * 3,
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmh2o",
        location="Barometric",
        sensor="Sensor 1",
    )


@pytest.fixture
def synthetic_expected_compensated_timeseries():
    """An expected compensated timeseries."""

    data = (
        pd.Series(
            [3.03, 3.02, 3.00],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        + 100
    )
    return Timeseries(
        ts=data,
        variable="head",
        unit="m asl",
        location="Station A",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def synthetic_timeseries_with_none():
    """A pd.Series used to create a Timeseries but containing non-float convertible values."""
    return pd.Series(
        [1313.25, 1312.5, None],
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )


@pytest.fixture
def synthetic_temperature_timeseries():
    """Create a temperature Timeseries."""
    data = pd.Series(
        [25, 30, 35], index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    )
    return Timeseries(
        ts=data,
        variable="temperature",
        unit="degc",
        location="Station B",
        sensor="Sensor 2",
    )


@pytest.fixture
def synthetic_dataset(synthetic_submerged_timeseries):
    """Create a Dataset with two Timeseries."""

    ts1 = synthetic_submerged_timeseries
    ts2 = ts1.model_copy(update={"location": "Station B", "sensor": "Sensor 2"})

    dataset = Dataset(timeseries=[ts1, ts2])

    return dataset


# ============================ Sample timeseries van Essen =============================


@pytest.fixture
def baro_timeseries() -> Timeseries:
    """Actual barometric pressure timeseries from a van Essen Diver.

    !!! note

        This test also relies on the read_from_csv function from gensor.getters module.
    """
    from gensor.testdata import baro

    return read_from_csv(path=baro, file_format="vanessen")


@pytest.fixture
def pb01a_timeseries() -> Timeseries:
    """Actual PB01A well timeseries from a van Essen Diver.

    !!! note

        This test also relies on the read_from_csv function from gensor.getters module.
    """
    from gensor.testdata import pb01a

    pb01a = read_from_csv(
        path=pb01a, file_format="vanessen", location="PB01A", sensor="AV319"
    )

    pb01a[0].sensor_alt = 31.48

    return pb01a


@pytest.fixture
def pb02a_plain_timeseries() -> Timeseries:
    """Actual PB02A well timeseries from a van Essen Diver with metadata removed to simulate a plain file.
    This test also relies on the read_from_csv function from gensor.getters module.
    """
    from gensor.testdata import pb02a_plain

    return read_from_csv(
        path=pb02a_plain, file_format="plain", location="PB02A", sensor="AV319"
    )


@pytest.fixture
def pb01a_fieldwork() -> list:
    return {"PB01A": ["2020-08-25"]}


@pytest.fixture
def plain_csv_file(tmp_path: Path) -> Path:
    PLAIN_CSV_CONTENT = """timestamp,pressure
    2024-01-01 00:00:00,1013
    2024-01-01 01:00:00,1012
    2024-01-01 02:00:00,1011
    """
    """Creates a temporary plain CSV file with timestamp and pressure columns."""
    csv_file = tmp_path / "timeseries.csv"
    with open(csv_file, "w") as f:
        f.write(PLAIN_CSV_CONTENT)
    return csv_file


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Creates an empty directory."""
    return tmp_path
