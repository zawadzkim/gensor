import pandas as pd
import pytest

from gensor.core.dataset import Coverage, Dataset
from gensor.core.timeseries import Timeseries
from gensor.io.read import read_from_sql


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


def test_filter_exclude_single_location(synthetic_dataset):
    """exclude= drops timeseries matching the given attribute."""
    result = synthetic_dataset.filter(exclude={"location": "Station A"})
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_filter_exclude_location_list(synthetic_dataset):
    """exclude= accepts a list (drop any location in the list)."""
    result = synthetic_dataset.filter(exclude={"location": ["Station A", "Station B"]})
    assert isinstance(result, Dataset)
    assert len(result) == 0


def test_filter_exclude_attribute_pair_is_anded(synthetic_dataset):
    """All conditions in exclude must match (AND): only Station B / Sensor 2 is dropped."""
    result = synthetic_dataset.filter(exclude={"location": "Station B", "sensor": "Sensor 2"})
    assert isinstance(result, Timeseries)
    assert result.location == "Station A"


def test_filter_include_and_exclude_combined(synthetic_dataset):
    """exclude= is applied on top of the include filters."""
    result = synthetic_dataset.filter(variable="pressure", exclude={"location": "Station A"})
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_getitem_by_index_still_returns_timeseries(synthetic_dataset):
    """Integer indexing keeps working and returns a Timeseries reference."""
    assert isinstance(synthetic_dataset[0], Timeseries)


def test_getitem_by_location_single(synthetic_dataset):
    """A location with one matching timeseries returns a Timeseries."""
    ts = synthetic_dataset["Station A"]
    assert isinstance(ts, Timeseries)
    assert ts.location == "Station A"


def test_getitem_by_location_multiple(synthetic_dataset):
    """A location with several timeseries returns a Dataset of them."""
    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(
            update={"variable": "temperature", "unit": "degc"}
        )
    )
    result = synthetic_dataset["Station A"]
    assert isinstance(result, Dataset)
    assert len(result) == 2
    assert {ts.variable for ts in result} == {"pressure", "temperature"}


def test_getitem_by_location_and_variable_tuple(synthetic_dataset):
    """(location, variable) tuple indexing narrows to a single Timeseries."""
    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(
            update={"variable": "temperature", "unit": "degc"}
        )
    )
    ts = synthetic_dataset["Station A", "pressure"]
    assert isinstance(ts, Timeseries)
    assert ts.location == "Station A"
    assert ts.variable == "pressure"


def test_getitem_tuple_with_unit(synthetic_dataset):
    """(location, variable, unit) tuple is supported too."""
    ts = synthetic_dataset["Station A", "pressure", "cmh2o"]
    assert isinstance(ts, Timeseries)
    assert ts.unit == "cmh2o"


def test_getitem_tuple_missing_raises_keyerror(synthetic_dataset):
    """A tuple with no match raises KeyError."""
    with pytest.raises(KeyError):
        synthetic_dataset["Station A", "conductivity"]


def test_getitem_by_location_missing_raises_keyerror(synthetic_dataset):
    """Indexing an unknown location raises KeyError (dict-like)."""
    with pytest.raises(KeyError):
        synthetic_dataset["Non-existent Station"]


def test_getitem_by_location_list(synthetic_dataset):
    """A list of locations returns a Dataset with the matching timeseries."""
    result = synthetic_dataset[["Station A", "Station B"]]
    assert isinstance(result, Dataset)
    assert len(result) == 2


def test_contains_location(synthetic_dataset):
    """`location in dataset` checks membership by location name."""
    assert "Station A" in synthetic_dataset
    assert "Station B" in synthetic_dataset
    assert "Non-existent Station" not in synthetic_dataset


def test_one_returns_single_timeseries(synthetic_dataset):
    """one() returns a single Timeseries when exactly one matches."""
    ts = synthetic_dataset.one(location="Station A")
    assert isinstance(ts, Timeseries)
    assert ts.location == "Station A"


def test_one_raises_when_multiple_match(synthetic_dataset):
    """one() raises when more than one timeseries matches."""
    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(
            update={"variable": "temperature", "unit": "degc"}
        )
    )
    with pytest.raises(ValueError):
        synthetic_dataset.one(location="Station A")


def test_one_raises_when_no_match(synthetic_dataset):
    """one() raises when nothing matches."""
    with pytest.raises(ValueError):
        synthetic_dataset.one(location="Non-existent Station")


def test_get_locations_is_unique(synthetic_dataset):
    """get_locations() de-duplicates (one entry per location, as documented)."""
    synthetic_dataset.add(
        synthetic_dataset[0].model_copy(
            update={"variable": "temperature", "unit": "degc"}
        )
    )
    locations = synthetic_dataset.get_locations()
    assert sorted(locations) == ["Station A", "Station B"]


def test_to_sql_handles_large_series(db):
    """to_sql() must persist series larger than SQLite's bound-variable limit."""
    n = 300_000  # exceeds SQLite's bound-variable limit for a single multi-row INSERT
    idx = pd.date_range("2020-01-01", periods=n, freq="s", tz="UTC")
    big = Timeseries(
        ts=pd.Series(range(n), index=idx, dtype=float),
        variable="pressure",
        unit="cmh2o",
        location="BigStation",
        sensor="S1",
    )

    Dataset(timeseries=[big]).to_sql(db)  # must not raise "too many SQL variables"

    reloaded = read_from_sql(db, load_all=True)
    ts = reloaded if isinstance(reloaded, Timeseries) else reloaded[0]
    assert len(ts.ts) == n


def test_to_sql_skips_empty_timeseries(db, synthetic_submerged_timeseries):
    """to_sql() must not crash on empty timeseries (their start/end are NaT)."""
    empty = synthetic_submerged_timeseries.model_copy(
        update={
            "ts": pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float),
            "location": "EmptyLoc",
        }
    )
    ds = Dataset(timeseries=[synthetic_submerged_timeseries, empty])

    ds.to_sql(db)  # should skip the empty one rather than raise on NaT.strftime

    reloaded = read_from_sql(db, load_all=True)
    locations = (
        reloaded.get_locations()
        if isinstance(reloaded, Dataset)
        else [reloaded.location]
    )
    assert "Station A" in locations
    assert "EmptyLoc" not in locations


def test_coverage_table(synthetic_dataset):
    """Dataset.coverage exposes a per-timeseries table."""
    coverage = synthetic_dataset.coverage
    assert isinstance(coverage, Coverage)
    assert list(coverage.table.columns) == Coverage.columns
    assert len(coverage.table) == len(synthetic_dataset)
    assert set(coverage.table["location"]) == {"Station A", "Station B"}
    assert (coverage.table["records"] > 0).all()


def test_coverage_plot_returns_fig_ax(synthetic_dataset):
    """Dataset.coverage.plot() returns a (fig, ax) with one row per location."""
    import matplotlib

    matplotlib.use("Agg")
    fig, ax = synthetic_dataset.coverage.plot()
    assert fig is not None
    assert len(ax.get_yticks()) == len(synthetic_dataset.get_locations())
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    pytest.main()
