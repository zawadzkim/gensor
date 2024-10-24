from pathlib import Path

from gensor.core.dataset import Dataset
from gensor.core.timeseries import Timeseries
from gensor.io.read import read_from_csv


def test_read_from_csv_no_files(empty_directory: Path):
    """Test that the function skips when no CSV files are present."""
    result = read_from_csv(empty_directory, file_format="plain")

    assert isinstance(result, Dataset)
    assert len(result.timeseries) == 0


def test_read_from_csv_with_files(plain_csv_file: Path):
    """Test that the function correctly loads a CSV file."""
    result = read_from_csv(
        plain_csv_file,
        file_format="plain",
        col_names=["timestamp", "pressure"],
        location="Station 1",
        sensor="Sensor A",
    )

    assert isinstance(result, Timeseries)

    assert result.variable == "pressure"


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
