import pandas as pd
import pydantic as pyd
import pytest

from gensor import compensate
from gensor.core.timeseries import Timeseries


def test_compensator_with_valid_data(
    synthetic_submerged_timeseries,
    synthetic_barometric_pressure_timeseries,
    synthetic_expected_compensated_timeseries,
):
    """Test compensator with valid timeseries inputs."""
    compensated = compensate(
        raw=synthetic_submerged_timeseries,
        barometric=synthetic_barometric_pressure_timeseries,
    )

    assert isinstance(compensated, Timeseries), (
        "Compensation should return a valid Timeseries."
    )
    assert compensated.variable == "head", (
        "Compensated Timeseries should have type 'head'."
    )
    assert compensated.unit == "m asl", (
        "Compensated Timeseries should have unit 'm asl'."
    )

    pd.testing.assert_series_equal(
        compensated.ts,
        synthetic_expected_compensated_timeseries.ts,
        check_exact=False,
        rtol=1e-5,
    )


def test_compensation_with_baro_as_float(
    synthetic_submerged_timeseries,
    synthetic_expected_compensated_timeseries,
    barometric_value,
):
    compensated = compensate(
        raw=synthetic_submerged_timeseries, barometric=barometric_value
    )

    assert isinstance(compensated, Timeseries), (
        "Compensation should return a valid Timeseries."
    )
    assert compensated.variable == "head", (
        "Compensated Timeseries should have type 'head'."
    )
    assert compensated.unit == "m asl", (
        "Compensated Timeseries should have unit 'm asl'."
    )

    pd.testing.assert_series_equal(
        compensated.ts,
        synthetic_expected_compensated_timeseries.ts,
        check_exact=False,
        rtol=1e-5,
    )


def test_invalid_raw_timeseries_type(
    synthetic_temperature_timeseries, barometric_value
):
    """Test that InvalidMeasurementTypeError is raised for wrong timeseries type."""

    with pytest.raises(pyd.ValidationError):
        compensate(raw=synthetic_temperature_timeseries, barometric=barometric_value)


def test_invalid_barometric_timeseries_type(
    synthetic_submerged_timeseries, synthetic_temperature_timeseries
):
    """Test that InvalidMeasurementTypeError is raised for wrong barometric timeseries type."""
    with pytest.raises(pyd.ValidationError):
        compensate(
            raw=synthetic_submerged_timeseries,
            barometric=synthetic_temperature_timeseries,
        )


def test_missing_sensor_alt(synthetic_submerged_timeseries, barometric_value):
    """Test that MissingInputError is raised when sensor_alt is missing."""
    missing_sensor_alt = synthetic_submerged_timeseries.model_copy(
        update={"sensor_alt": None}
    )
    with pytest.raises(pyd.ValidationError):
        compensate(raw=missing_sensor_alt, barometric=barometric_value)


def test_mask_fieldwork_days(pb01a_fieldwork, pb01a_timeseries, baro_timeseries):
    """Test removal of erroneous measurements with a mask of fieldwork events"""

    comp_ts = compensate(
        raw=pb01a_timeseries,
        barometric=baro_timeseries[0],
        fieldwork_dates=pb01a_fieldwork,
    )

    assert len(comp_ts.ts) == len(pb01a_timeseries.ts)
