import os
from pathlib import Path

import pandas as pd
import pytest

from gensor import Timeseries, read_from_csv
from gensor.db import DatabaseConnection


@pytest.fixture
def db(tmp_path):
    """Fixture to create a temporary database file for each test."""
    db_connection = DatabaseConnection(db_directory=tmp_path, db_name="test_db.sqlite")

    yield db_connection

    db_path = tmp_path / "test_db.sqlite"
    if db_path.exists():
        os.remove(db_path)


@pytest.fixture
def simple_timeseries():
    """Create a simple Timeseries for testing."""
    data = pd.Series(
        [1.0, 2.0, 3.0],
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
def timeseries_with_datetime():
    file_path = Path("tests/.data/temperature_data_with_time.csv")
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    df.index = df.index.tz_localize("UTC")

    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station1",
        sensor="Sensor123",
        unit="degc",
    )


@pytest.fixture
def timeseries_with_datetime_other():
    file_path = Path("tests/.data/temperature_data_with_time_other.csv")
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    df.index = df.index.tz_localize("UTC")

    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station1",
        sensor="Sensor123",
        unit="degc",
    )


@pytest.fixture
def timeseries_with_date():
    file_path = Path("tests/.data/temperature_data_without_time.csv")
    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    df.index = df.index.tz_localize("UTC")
    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station2",
        sensor="Sensor456",
        unit="degc",
    )


# ======================== For compensation test =======================================


@pytest.fixture
def valid_submerged_timeseries():
    """Create a valid Timeseries instance with sensor altitude."""
    data = pd.Series(
        [1313.25, 1312.5, 1310.0],
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
def valid_barometric_timeseries():
    """Create a valid Timeseries instance with sensor altitude."""
    data = pd.Series(
        [1013.25, 1012.5, 1010.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmh2o",
        location="Barometric",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def expected_compensated_timeseries():
    """Create an expected compensated timeseries."""

    data = (
        pd.Series(
            [3.0, 3.0, 3.0],
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


# ======================== For compensation test =======================================


@pytest.fixture
def invalid_timeseries_type():
    """Create a Timeseries instance with the wrong type."""
    data = pd.Series(
        [25, 30, 35], index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    )
    return Timeseries(
        ts=data,
        variable="temperature",
        unit="degc",
        location="Station B",
        sensor="Sensor 2",
        sensor_alt=200,
    )


@pytest.fixture
def missing_sensor_alt_timeseries():
    """Create a Timeseries instance with no sensor_alt."""
    data = pd.Series(
        [1013.25, 1012.5, 1010.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmh2o",
        location="Station A",
        sensor="Sensor 1",
    )


@pytest.fixture
def timeseries() -> Timeseries:
    """Actual barometric pressure timeseries from a van Essen Diver. This test also
    relies on the read_from_csv function from gensor.getters module.
    """
    file_path = Path("tests/.data/BY222_Barodiver_TEST.CSV")
    return read_from_csv(path=file_path, file_format="vanessen")
