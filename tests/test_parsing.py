from gensor import Dataset, Timeseries


def test_parsing_vanessen(baro_timeseries):
    dataset = baro_timeseries

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2, "There should be two timeseties in the test file."
    assert isinstance(dataset[0], Timeseries)
    assert dataset[0].location == "Barodiver", "The test file is data \
        from a station named 'Barodiver'"
    assert dataset[0].variable == "pressure", "The first object should be the \
          pressure measurements"


def test_plain_parser(pb02a_plain_timeseries):
    ds = pb02a_plain_timeseries

    assert isinstance(ds, Dataset), "The resulting object should be a Dataset."
