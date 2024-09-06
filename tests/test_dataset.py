import pytest

from gensor import Dataset


def test_add_timeseries_to_dataset(simple_timeseries, timeseries):
    """Try instanciating an empty Dataset, then add one timeseries to it. Next, check
    if adding the same Timeseries will not create an extra Timeseries. Finally check if
    adding another timeseries will raise Dataset len to 2."""

    ds = Dataset()
    assert len(ds) == 0, "This should return an info that the ds is empty"

    ds.add(simple_timeseries)
    assert len(ds) == 1, "This should be equal to 1 after adding a ts."

    ds.add(simple_timeseries)
    assert len(ds) == 1, "This still should be 1, because the same timeseries is added"

    ds.add(timeseries[0])
    assert len(ds) == 2, "This should be 2, because another timeseries is added"


def test_add_two_datasets(timeseries):
    """Add a dataset from Barodiver to an empty dataset and check if there are two
    timeseries (temp, press)"""
    ds = Dataset()

    ds.add(timeseries)
    assert (
        len(ds) == 2
    ), "This still should be 2, because Dataset(2) was added to Dataset(0)"


if __name__ == "__main__":
    pytest.main()
