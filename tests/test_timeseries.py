"""Testing the Timeseries object creation and functionality:

- test_create_timeseries: Check if Timeseries is correctly created.
- test_coercing_datetimeindex: Check if provided date index is coerced to datetime.
- test_timeseries_invalid_schema: Check if an error is raised if the timeseries contains non-float convertible values.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from gensor import Timeseries

# ============================== Test Timeseries creation ==============================


def test_create_timeseries(synthetic_submerged_timeseries):
    """Check if Timeseries is correctly created."""

    ts = synthetic_submerged_timeseries

    assert isinstance(ts.ts, pd.Series)
    assert ts.variable == "pressure"
    assert isinstance(
        ts.ts.index, pd.DatetimeIndex
    ), "Index is not of datetime type"
    assert ts.start == pd.Timestamp(
        "2024-01-01 00:00:00+00:00")
    assert ts.end == pd.Timestamp(
        "2024-01-01 02:00:00+00:00")


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


def test_adding_timeseries(synthetic_submerged_timeseries,
                           extended_synthetic_submerged_pressure_timeseries):
    new_ts = synthetic_submerged_timeseries.concatenate(
        extended_synthetic_submerged_pressure_timeseries)

    # The new Series should be exactly 3 records longer
    assert (
        len(new_ts.ts) == len(synthetic_submerged_timeseries.ts) + 3
    ), f"Expected new timeseries to have 6 records, but got {len(new_ts.ts)}"


def test_adding_timeseries_not_equal(synthetic_submerged_timeseries):
    unequal_serie = synthetic_submerged_timeseries.model_copy(
        update={"location": "Station2"})

    with pytest.raises(ValueError):
        synthetic_submerged_timeseries.concatenate(unequal_serie)


if __name__ == "__main__":
    pytest.main()
