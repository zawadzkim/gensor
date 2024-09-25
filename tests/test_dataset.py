import pytest

from gensor import Dataset


def test_add_timeseries_to_dataset(pb01a_timeseries, baro_timeseries):
    """Test creating dataset and adding datasets together."""

    ds = Dataset()
    assert len(ds) == 0, "This should return an info that the ds is empty"

    ds.add(pb01a_timeseries[0])
    assert len(ds) == 1, "This should be equal to 1 after adding a ts."

    ds.add(pb01a_timeseries[0])
    assert len(ds) == 1, "This still should be 1, because the same timeseries is added"

    ds.add(baro_timeseries[0])
    assert len(ds) == 2, "This should be 2, because another timeseries is added"


if __name__ == "__main__":
    pytest.main()
