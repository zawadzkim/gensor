from pathlib import Path

import pandas as pd
import pytest

from gensor import Dataset, Timeseries

file_path = Path("tests/.data/temperature_data_without_time.csv")
df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

ts = df["temperature"]
timeseries = Timeseries(
    ts=ts, variable="temperature", location="Station2", sensor="Sensor456", unit="degC"
)


def test_empty_dataset():
    ds = Dataset()

    assert len(ds) == 0, "This should return an info that the ds is empty"


def test_add_timeseries_to_dataset():
    ds = Dataset()
    ds.add(timeseries)
    assert len(ds) == 1, "This should be equal to 1 after adding a ts."

    ds.add(timeseries)
    assert len(ds) == 1, "This still should be 1, because the same timeseries \
        is added"


if __name__ == "__main__":
    pytest.main()
