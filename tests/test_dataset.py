import pytest

from gensor.core.dataset import Dataset
from gensor.core.timeseries import Timeseries


def test_add_timeseries_to_dataset(pb01a_timeseries, baro_timeseries):
    """Test creating dataset and adding datasets together."""

    ds = Dataset()
    assert len(ds) == 0, "This should return an info that the ds is empty"

    ds.add(pb01a_timeseries)
    assert len(ds) == 1, "This should be equal to 1 after adding a ts."

    ds.add(pb01a_timeseries)
    assert len(ds) == 1, "This still should be 1, because the same timeseries is added"

    ds.add(baro_timeseries[0])
    assert len(ds) == 2, "This should be 2, because another timeseries is added"


def test_filter_by_station_and_variable(synthetic_dataset):
    """Test filtering the dataset by station and variable."""
    dataset = synthetic_dataset

    single_ts = dataset.filter(location="Station A", variable="pressure", unit="cmh2o")

    assert isinstance(single_ts, Timeseries), (
        "If only one ts is found, the return should be a Timeseries, not Dataset(1)"
    )

    assert single_ts.location == "Station A"
    assert single_ts.variable == "pressure"
    assert single_ts.unit == "cmh2o"


def test_filter_multiple_results(synthetic_dataset):
    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(
            update={"location": "Station C", "variable": "temperature"}
        )
    )

    two_ts_ds = synthetic_dataset.filter(variable="pressure", unit="cmh2o")

    assert len(two_ts_ds) == 2, "There are two pressure timeseries in this dataset."


def test_filter_no_match(synthetic_dataset):
    """Test filtering the dataset with no matching criteria."""
    dataset = synthetic_dataset

    filtered = dataset.filter(
        location="Non-existent Station", variable="pressure", unit="cmh2o"
    )
    assert len(filtered) == 0


def test_filter_by_station_and_sensor(synthetic_dataset):
    """Test filtering the dataset by station and sensor."""

    single_ts = synthetic_dataset.filter(location="Station B", sensor="Sensor 2")

    assert isinstance(single_ts, Timeseries), (
        "If only one ts is found, the return should be a Timeseries, not Dataset(1)"
    )

    assert single_ts.location == "Station B"
    assert single_ts.sensor == "Sensor 2"
    assert single_ts.variable == "pressure"
    assert single_ts.unit == "cmh2o"


def test_filter_missing_attribute(synthetic_dataset):
    try:
        synthetic_dataset.filter(
            location="Station A",
            variable="pressure",
            unit="cmh2o",
            non_exising_attr="Foo",
        )
    except AttributeError as e:
        assert str(e) == "'Timeseries' object has no attribute 'non_exising_attr'"


def test_filter_with_attribute_as_list(synthetic_dataset):
    """Test filtering the dataset when the attribute values are passed as lists."""

    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(update={"sensor": "Sensor 3"})
    )

    filtered_dataset = synthetic_dataset.filter(
        location="Station A",
        variable="pressure",
        unit="cmh2o",
        sensor=["Sensor 1", "Sensor 3"],
    )

    assert len(filtered_dataset) == 2, (
        "There should be two timeseries with Sensor 1 and Sensor 3"
    )

    sensors_in_result = {ts.sensor for ts in filtered_dataset}
    assert "Sensor 1" in sensors_in_result
    assert "Sensor 3" in sensors_in_result


if __name__ == "__main__":
    pytest.main()
