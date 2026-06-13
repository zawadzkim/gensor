import pandas as pd
import pytest

from gensor.core.dataset import Coverage, CoverageDiff, Dataset, Where, diff
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


def test_filter_negate_single_location(synthetic_dataset):
    """A ``~``-prefixed value drops timeseries with that attribute value."""
    result = synthetic_dataset.filter(location="~Station A")
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_filter_negate_location_list(synthetic_dataset):
    """A list of ``~``-prefixed values drops any location in the list."""
    result = synthetic_dataset.filter(location=["~Station A", "~Station B"])
    assert isinstance(result, Dataset)
    assert len(result) == 0


def test_filter_negate_by_kwarg(synthetic_dataset):
    """Negation works on keyword attributes too (e.g. drop a single sensor)."""
    result = synthetic_dataset.filter(sensor="~Sensor 2")
    assert isinstance(result, Timeseries)
    assert result.location == "Station A"


def test_filter_include_and_negate_combined(synthetic_dataset):
    """Positive and negated filters are AND-ed across attributes."""
    result = synthetic_dataset.filter(variable="pressure", location="~Station A")
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_filter_mixed_include_and_negate_one_attribute(synthetic_dataset):
    """Positive and negated values may be mixed within a single attribute."""
    result = synthetic_dataset.filter(location=["Station A", "Station B", "~Station A"])
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_filter_with_where_predicate_negated_combination(synthetic_dataset):
    """filter() accepts Where predicates; ~Where(a, b) drops only the (a AND b) series."""
    result = synthetic_dataset.filter(~Where(location="Station A", sensor="Sensor 1"))
    assert isinstance(result, Timeseries)
    assert result.location == "Station B"


def test_pop_removes_and_returns(synthetic_dataset):
    """pop() returns the matching timeseries and removes it from the dataset."""
    n = len(synthetic_dataset)
    popped = synthetic_dataset.pop(location="Station A")
    assert isinstance(popped, Timeseries)
    assert popped.location == "Station A"
    assert len(synthetic_dataset) == n - 1
    assert "Station A" not in synthetic_dataset


def test_pop_returns_reference_for_roundtrip(synthetic_dataset):
    """pop() returns the live object (not a copy); edit then add() round-trips."""
    ts = synthetic_dataset.pop(location="Station A")
    ts.ts = ts.ts * 0 + 42.0
    synthetic_dataset.add(ts)
    assert (synthetic_dataset["Station A"].ts == 42.0).all()


def test_loc_slices_every_timeseries(pb01a_timeseries, baro_timeseries):
    """ds.loc[start:end] slices each timeseries by label and returns a new Dataset."""
    ds = Dataset(timeseries=[pb01a_timeseries, baro_timeseries[0]])
    start, end = "2021-01-01", "2021-06-30"

    sub = ds.loc[start:end]

    assert isinstance(sub, Dataset)
    assert len(sub) == len(ds)
    for ts in sub:
        assert str(ts.ts.index.min().date()) >= start
        assert str(ts.ts.index.max().date()) <= end
    # original is untouched
    assert pb01a_timeseries.ts.index.min() < pd.Timestamp(start, tz="UTC")


def test_loc_scalar_key_raises(synthetic_dataset):
    """A point lookup (scalar per series) is rejected with a clear message."""
    ts0 = synthetic_dataset[0].ts.index[0]
    with pytest.raises(TypeError, match="label slice"):
        synthetic_dataset.loc[ts0]


def test_pop_no_match_removes_nothing(synthetic_dataset):
    """pop() with no match returns an empty Dataset and leaves the dataset unchanged."""
    n = len(synthetic_dataset)
    result = synthetic_dataset.pop(location="Nonexistent")
    assert isinstance(result, Dataset)
    assert len(result) == 0
    assert len(synthetic_dataset) == n


def test_filter_negated_combination_spares_partial_match(synthetic_dataset):
    """~Where(a, b) must NOT drop series that match only some of the combined conditions."""
    # No series is (Station A AND Sensor 2), so the negated combination drops nothing.
    result = synthetic_dataset.filter(~Where(location="Station A", sensor="Sensor 2"))
    assert isinstance(result, Dataset)
    assert len(result) == 2


def test_filter_union_of_negated_predicates(synthetic_dataset):
    """Several negated predicates AND together to remove the union of their matches."""
    result = synthetic_dataset.filter(~Where(location="Station A"), ~Where(location="Station B"))
    assert isinstance(result, Dataset)
    assert len(result) == 0


def test_where_or_combination(synthetic_dataset):
    """Where supports | (or): keep series matching either branch."""
    result = synthetic_dataset.filter(Where(location="Station A") | Where(location="Station B"))
    assert isinstance(result, Dataset)
    assert len(result) == 2


def test_where_and_with_keyword_filter(synthetic_dataset):
    """A positional Where is AND-ed with the keyword filters in the same call."""
    result = synthetic_dataset.filter(~Where(location="Station A"), variable="pressure")
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


def test_info_table(synthetic_dataset):
    """Dataset.info is a per-timeseries metadata table."""
    info = synthetic_dataset.info
    assert isinstance(info, pd.DataFrame)
    assert list(info.columns) == ["location", "variable", "sensor", "records", "start", "end"]
    assert len(info) == len(synthetic_dataset)
    assert set(info["location"]) == {"Station A", "Station B"}
    assert (info["records"] > 0).all()


def test_coverage_plot_returns_fig_ax(synthetic_dataset):
    """Dataset.coverage.plot() returns a (fig, ax) with one row per location."""
    import matplotlib

    matplotlib.use("Agg")
    fig, ax = synthetic_dataset.coverage.plot()
    assert fig is not None
    assert len(ax.get_yticks()) == len(synthetic_dataset.get_locations())
    matplotlib.pyplot.close(fig)


def test_diff_table_and_status(synthetic_dataset):
    """Dataset.diff aligns by (location, variable) and reports per-dataset coverage."""
    other = Dataset(timeseries=[synthetic_dataset[0].model_copy(deep=True)])  # Station A only
    result = synthetic_dataset.diff(other, labels=["full", "partial"])

    assert isinstance(result, CoverageDiff)
    for col in [("full", "records"), ("partial", "records"), ("summary", "status"),
                ("summary", "present")]:
        assert col in result.table.columns

    status = {idx[0]: val for idx, val in result.table[("summary", "status")].items()}
    present = {idx[0]: val for idx, val in result.table[("summary", "present")].items()}
    assert status["Station A"] == "identical"
    assert status["Station B"] == "only full"
    assert present["Station A"] == 2
    assert present["Station B"] == 1


def test_diff_module_function_with_list_autolabels(synthetic_dataset):
    """diff() accepts a list and auto-labels ds0, ds1, ..."""
    other = Dataset(timeseries=[synthetic_dataset[0].model_copy(deep=True)])
    result = diff([synthetic_dataset, other])
    assert ("ds0", "records") in result.table.columns
    assert ("ds1", "records") in result.table.columns


def test_diff_requires_at_least_two(synthetic_dataset):
    with pytest.raises(ValueError):
        diff({"only": synthetic_dataset})


def test_diff_plot_returns_fig_ax(synthetic_dataset):
    """CoverageDiff.plot() returns a (fig, ax) with one row per aligned key."""
    import matplotlib

    matplotlib.use("Agg")
    other = Dataset(timeseries=[synthetic_dataset[0].model_copy(deep=True)])
    comparison = synthetic_dataset.diff(other, labels=["a", "b"])
    fig, ax = comparison.plot()
    assert fig is not None
    assert len(ax.get_yticks()) == len(comparison.keys)  # one row per aligned (location, variable)
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    pytest.main()
