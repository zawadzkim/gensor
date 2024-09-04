"""TODO: implement test for saving to sql"""

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from gensor import Timeseries


@pytest.fixture
def simple_timeseries():
    """Create a simple Timeseries for testing."""
    data = pd.Series(
        [1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="h")
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmH2O",
        location="Station A",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def timeseries_with_datetime():
    file_path = Path("tests/.data/temperature_data_with_time.csv")
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station1",
        sensor="Sensor123",
        unit="degC",
    )


@pytest.fixture
def timeseries_with_datetime_other():
    file_path = Path("tests/.data/temperature_data_with_time_other.csv")
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station1",
        sensor="Sensor123",
        unit="degC",
    )


@pytest.fixture
def timeseries_with_date():
    file_path = Path("tests/.data/temperature_data_without_time.csv")
    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

    ts = df["temperature"]
    return Timeseries(
        ts=ts,
        variable="temperature",
        location="Station2",
        sensor="Sensor456",
        unit="degC",
    )


def test_timeseries_with_time(timeseries_with_datetime):
    assert isinstance(timeseries_with_datetime.ts, pd.Series)
    assert timeseries_with_datetime.variable == "temperature"
    assert isinstance(
        timeseries_with_datetime.ts.index, pd.DatetimeIndex
    ), "Index is not of datetime type"


def test_timeseries_with_date(timeseries_with_date):
    assert isinstance(timeseries_with_date.ts, pd.Series)
    assert timeseries_with_date.variable == "temperature"
    assert isinstance(
        timeseries_with_date.ts.index, pd.DatetimeIndex
    ), "Index is not of datetime type"


def test_timeseries_invalid_schema():
    invalid_series = pd.Series(
        ["a", "b", "c"],
        index=pd.to_datetime(["2024-09-01", "2024-09-02", "2024-09-03"]),
    )

    with pytest.raises(SchemaError):
        Timeseries(
            ts=invalid_series,
            variable="temperature",
            location="Station3",
            sensor="Sensor789",
            unit="degC",
        )


def test_adding_timeseries(timeseries_with_datetime, timeseries_with_datetime_other):
    new_ts = timeseries_with_datetime.concatenate(timeseries_with_datetime_other)

    # The new Series should be exactly 5 records longer
    assert (
        len(new_ts.ts) == len(timeseries_with_datetime.ts) + 5
    ), f"Expected new timeseries to have {len(timeseries_with_datetime.ts) + 5} \
       records, but got {len(new_ts.ts)}"


def test_adding_timeseries_not_equal(timeseries_with_datetime):
    unequal_serie = timeseries_with_datetime.model_copy(update={"location": "Station2"})

    with pytest.raises(ValueError):
        timeseries_with_datetime.concatenate(unequal_serie)


if __name__ == "__main__":
    pytest.main()
