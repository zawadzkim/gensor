from pathlib import Path

from gensor import Dataset, Timeseries, read_from_csv


def test_parsing_vanessen():
    file_path = Path("tests/.data/BY222_Barodiver_TEST.CSV")

    dataset = read_from_csv(path=file_path, file_format="vanessen")

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2, "There should be two timeseties in the test file."
    assert isinstance(dataset[0], Timeseries)
    assert dataset[0].location == "Barodiver", "The test file is data \
        from a station named 'Baridiver'"
    assert dataset[0].variable == "pressure", "The first object should be the \
          pressure measurements"
