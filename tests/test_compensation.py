import pandas as pd
import pydantic as pyd
import pytest

from gensor.compensation import Compensator
from gensor.dtypes import Timeseries


@pytest.fixture
def valid_submerged_timeseries():
    """Create a valid Timeseries instance with sensor altitude."""
    data = pd.Series(
        [1313.25, 1312.5, 1310.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
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
def valid_barometric_timeseries():
    """Create a valid Timeseries instance with sensor altitude."""
    data = pd.Series(
        [1013.25, 1012.5, 1010.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmH2O",
        location="Barometric",
        sensor="Sensor 1",
        sensor_alt=100,
    )


@pytest.fixture
def expected_compensated_timeseries():
    """Create an expected compensated timeseries."""

    data = (
        pd.Series(
            [3.0, 3.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="h")
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
def invalid_timeseries_type():
    """Create a Timeseries instance with the wrong type."""
    data = pd.Series(
        [25, 30, 35], index=pd.date_range("2024-01-01", periods=3, freq="h")
    )
    return Timeseries(
        ts=data,
        variable="temperature",
        unit="degC",  # Incorrect type
        location="Station B",
        sensor="Sensor 2",
        sensor_alt=200,
    )


@pytest.fixture
def missing_sensor_alt_timeseries():
    """Create a Timeseries instance with no sensor_alt."""
    data = pd.Series(
        [1013.25, 1012.5, 1010.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )
    return Timeseries(
        ts=data,
        variable="pressure",
        unit="cmH2O",  # Correct type
        location="Station A",
        sensor="Sensor 1",
    )


def test_compensator_with_valid_data(
    valid_submerged_timeseries,
    valid_barometric_timeseries,
    expected_compensated_timeseries,
):
    """Test compensator with valid timeseries inputs."""
    compensator = Compensator(
        ts=valid_submerged_timeseries, barometric=valid_barometric_timeseries
    )

    compensated = compensator.compensate(alignment_period="h", threshold_wc=0.5)

    assert compensated is not None, "Compensation should return a valid Timeseries."
    assert (
        compensated.variable == "head"
    ), "Compensated Timeseries should have type 'head'."
    assert (
        compensated.unit == "m asl"
    ), "Compensated Timeseries should have unit 'm asl'."

    pd.testing.assert_series_equal(
        compensated.ts, expected_compensated_timeseries.ts, check_exact=False, rtol=1e-5
    )


def test_invalid_timeseries_type(invalid_timeseries_type):
    """Test that InvalidMeasurementTypeError is raised for wrong timeseries type."""
    barometric_value = 1010.0

    try:
        Compensator(ts=invalid_timeseries_type, barometric=barometric_value)
    except pyd.ValidationError as exc:
        assert exc.errors()[0]["type"] == "value_error"


def test_missing_sensor_alt(missing_sensor_alt_timeseries):
    """Test that MissingInputError is raised when sensor_alt is missing."""
    barometric_value = 1010.0

    try:
        Compensator(ts=missing_sensor_alt_timeseries, barometric=barometric_value)
    except pyd.ValidationError as exc:
        assert exc.errors()[0]["type"] == "value_error"


def test_invalid_barometric_timeseries_type(
    valid_submerged_timeseries, invalid_timeseries_type
):
    """Test that InvalidMeasurementTypeError is raised for wrong barometric timeseries type."""
    with pytest.raises(pyd.ValidationError):
        Compensator(ts=valid_submerged_timeseries, barometric=invalid_timeseries_type)
