import pandas as pd
import pydantic as pyd
import pytest

from gensor.compensation import Compensator


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
