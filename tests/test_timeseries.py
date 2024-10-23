"""Testing the Timeseries object creation and functionality:

- test_create_timeseries: Check if Timeseries is correctly created.
- test_coercing_datetimeindex: Check if provided date index is coerced to datetime.
- test_timeseries_invalid_schema: Check if an error is raised if the timeseries contains non-float convertible values.
"""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from gensor.core.timeseries import Timeseries

# ============================== Test Timeseries creation ==============================


def test_create_timeseries(synthetic_submerged_timeseries):
    """Check if Timeseries is correctly created."""

    ts = synthetic_submerged_timeseries

    assert isinstance(ts.ts, pd.Series)
    assert ts.variable == "pressure"
    assert isinstance(ts.ts.index, pd.DatetimeIndex), "Index is not of datetime type"
    assert ts.start == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert ts.end == pd.Timestamp("2024-01-01 02:00:00+00:00")


def test_timeseries_invalid_schema(synthetic_timeseries_with_none):
    """Check if an error is raised if the timeseries contains non-float convertible values."""

    with pytest.raises(SchemaError):
        Timeseries(
            ts=synthetic_timeseries_with_none,
            variable="pressure",
            unit="cmh2o",
            location="Station A",
            sensor="Sensor 1",
            sensor_alt=100,
        )


# ============================= Test Timeseries functions ==============================


def test_adding_timeseries(
    synthetic_submerged_timeseries, extended_synthetic_submerged_pressure_timeseries
):
    new_ts = synthetic_submerged_timeseries.concatenate(
        extended_synthetic_submerged_pressure_timeseries
    )

    # The new Series should be exactly 3 records longer
    assert (
        len(new_ts.ts) == len(synthetic_submerged_timeseries.ts) + 3
    ), f"Expected new timeseries to have 6 records, but got {len(new_ts.ts)}"


def test_adding_timeseries_not_equal(synthetic_submerged_timeseries):
    unequal_serie = synthetic_submerged_timeseries.model_copy(
        update={"location": "Station2"}
    )

    with pytest.raises(ValueError):
        synthetic_submerged_timeseries.concatenate(unequal_serie)


def test_mask_with(
    synthetic_submerged_timeseries, extended_synthetic_submerged_pressure_timeseries
):
    """
    Test the mask_with method which filters a Timeseries to only include records
    whose indices are present in another Timeseries.
    """

    overlap_data = pd.Series(
        [1312.5], index=[pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC")]
    )
    extended_synthetic_submerged_pressure_timeseries.ts = pd.concat([
        extended_synthetic_submerged_pressure_timeseries.ts,
        overlap_data,
    ])

    masked_ts = synthetic_submerged_timeseries.mask_with(
        extended_synthetic_submerged_pressure_timeseries, mode="keep"
    )

    expected_index = pd.Index([pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC")])

    assert masked_ts.ts.index.equals(expected_index), "Masked Timeseries index mismatch"

    assert len(synthetic_submerged_timeseries.ts) > len(
        masked_ts.ts
    ), "Original timeseries was altered"


# =================================== Test indexing ====================================


def test_loc_single_timestamp(synthetic_submerged_timeseries):
    """Test the loc indexer for a single timestamp."""
    result = synthetic_submerged_timeseries.loc["2024-01-01 01:00:00+00:00"]
    assert isinstance(result, np.float64)
    assert result == 1312.00


def test_loc_timestamp_range(synthetic_submerged_timeseries):
    """Test the loc indexer for a timestamp range."""
    result = synthetic_submerged_timeseries.loc[
        "2024-01-01 01:00:00+00:00":"2024-01-01 02:00:00+00:00"
    ]
    assert isinstance(result, Timeseries)
    pd.testing.assert_series_equal(
        result.ts,
        synthetic_submerged_timeseries.ts.loc[
            "2024-01-01 01:00:00+00:00":"2024-01-01 02:00:00+00:00"
        ],
    )


def test_iloc_single_position(synthetic_submerged_timeseries):
    """Test the iloc indexer for a single positional index."""
    result = synthetic_submerged_timeseries.iloc[2]
    assert isinstance(result, np.float64)
    assert result == 1310.00


def test_iloc_range(synthetic_submerged_timeseries):
    """Test the iloc indexer for a positional range."""
    result = synthetic_submerged_timeseries.iloc[0:2]
    assert isinstance(result, Timeseries)
    pd.testing.assert_series_equal(
        result.ts, synthetic_submerged_timeseries.ts.iloc[0:2]
    )


# =================================== Test equality ====================================
def test_equal_timeseries(synthetic_submerged_timeseries):
    data = synthetic_submerged_timeseries
    other = synthetic_submerged_timeseries.model_copy(deep=True)

    assert data == other


def test_unequal_timeseries(synthetic_submerged_timeseries):
    data = synthetic_submerged_timeseries
    other = synthetic_submerged_timeseries.model_copy(
        deep=True, update={"variable": "temperature"}
    )

    assert data != other


if __name__ == "__main__":
    pytest.main()
