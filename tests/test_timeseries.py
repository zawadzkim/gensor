import pandas as pd
import pytest
from pandera.errors import SchemaError

from gensor import Timeseries


def test_timeseries_with_time(timeseries_with_datetime):
    assert isinstance(timeseries_with_datetime.ts, pd.Series)
    assert timeseries_with_datetime.variable == "temperature"
    assert isinstance(
        timeseries_with_datetime.ts.index, pd.DatetimeIndex
    ), "Index is not of datetime type"
    assert timeseries_with_datetime.start == pd.Timestamp("2024-09-01 00:00:00+00:00")
    assert timeseries_with_datetime.end == pd.Timestamp("2024-09-03 01:00:00+00:00")


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
