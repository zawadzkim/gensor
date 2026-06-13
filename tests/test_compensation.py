import pandas as pd
import pydantic as pyd
import pytest

from gensor import compensate, water_column
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


def test_threshold_wc_drops_negative_and_near_zero_water_columns():
    """threshold_wc must drop near-zero AND negative water columns (out-of-water /
    erroneous), not just those whose absolute value is small."""
    idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
    # baro = 1000 cmH2O; water columns (m): +5.0, +4.5, -0.02 (near zero), -0.5 (erroneous), +4.8
    sensor = pd.Series([1500.0, 1450.0, 998.0, 950.0, 1480.0], index=idx)
    ts = Timeseries(
        ts=sensor,
        variable="pressure",
        unit="cmh2o",
        location="PBX",
        sensor="S1",
        sensor_alt=30.0,
    )

    compensated = compensate(raw=ts, barometric=1000.0, threshold_wc=0.05)

    # only the three positive-above-cutoff water columns survive (heads 35.0, 34.5, 34.8)
    assert len(compensated.ts) == 3
    assert (compensated.ts >= 30.0).all()
    # the erroneous -0.5 m water column (head 29.5) must be gone
    assert 29.5 not in set(compensated.ts.round(3))
    # both the near-zero and the negative reading are recorded as dropped outliers
    assert len(compensated.outliers) == 2


def test_water_column_standalone():
    """water_column() returns just the compensated water column (no sensor altitude)."""
    idx = pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC")
    # baro = 1000 cmH2O; water columns (m): +5.0, +0.01 (< 25 mm), -0.3, +4.0
    sensor = pd.Series([1500.0, 1001.0, 970.0, 1400.0], index=idx)
    ts = Timeseries(
        ts=sensor,
        variable="pressure",
        unit="cmh2o",
        location="PBX",
        sensor="S1",
        sensor_alt=30.0,
    )

    wc = water_column(raw=ts, barometric=1000.0)

    assert wc.variable == "water_column"
    assert wc.unit == "m"
    # same out-of-water filtering as compensate: only +5.0 and +4.0 survive
    assert set(wc.ts.round(2)) == {5.0, 4.0}
    assert len(wc.outliers) == 2


def test_compensate_equals_water_column_plus_sensor_alt():
    """compensate() head is exactly the water column plus the sensor altitude."""
    idx = pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC")
    sensor = pd.Series([1500.0, 1001.0, 970.0, 1400.0], index=idx)
    ts = Timeseries(
        ts=sensor,
        variable="pressure",
        unit="cmh2o",
        location="PBX",
        sensor="S1",
        sensor_alt=30.0,
    )

    wc = water_column(raw=ts, barometric=1000.0)
    head = compensate(raw=ts, barometric=1000.0)

    assert head.variable == "head"
    assert head.unit == "m asl"
    pd.testing.assert_series_equal(head.ts, wc.ts + 30.0, check_names=False)


def test_default_threshold_drops_negative_water_columns():
    """With no threshold_wc passed, the 25 mm default applies and drops negatives."""
    idx = pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC")
    # baro = 1000 cmH2O; water columns (m): +5.0, +0.01 (< 25 mm), -0.3 (erroneous), +4.0
    sensor = pd.Series([1500.0, 1001.0, 970.0, 1400.0], index=idx)
    ts = Timeseries(
        ts=sensor,
        variable="pressure",
        unit="cmh2o",
        location="PBX",
        sensor="S1",
        sensor_alt=30.0,
    )

    compensated = compensate(raw=ts, barometric=1000.0)  # default threshold_wc = 0.025

    # only +5.0 and +4.0 m survive (heads 35.0, 34.0); +0.01 and -0.3 are dropped
    assert len(compensated.ts) == 2
    assert set(compensated.ts.round(2)) == {35.0, 34.0}
    assert len(compensated.outliers) == 2


def test_mask_fieldwork_days(pb01a_fieldwork, pb01a_timeseries, baro_timeseries):
    """Fieldwork days are masked to NaN gaps and kept (not dropped) in the output."""

    comp_ts = compensate(
        raw=pb01a_timeseries,
        barometric=baro_timeseries[0],
        fieldwork_dates=pb01a_fieldwork,
    )

    # the fieldwork day (2020-08-25) survives as NaN gaps rather than being removed
    fieldwork_day = comp_ts.ts.loc["2020-08-25"]
    assert len(fieldwork_day) > 0
    assert fieldwork_day.isna().all()
