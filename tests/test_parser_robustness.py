"""Tests for van Essen parser robustness and explicit metadata overrides."""

from __future__ import annotations

from pathlib import Path

from gensor.core.dataset import Dataset
from gensor.io.read import read_from_csv
from gensor.parse.utils import detect_encoding
from gensor.testdata import pb01a


def _read_text(path: Path) -> tuple[str, str]:
    enc = detect_encoding(path, num_bytes=10_000)
    return Path(path).read_text(encoding=enc), enc


def _as_timeseries(result):
    return result[0] if isinstance(result, Dataset) else result


def test_vanessen_parses_without_end_marker(tmp_path):
    """Diver-Office exports sometimes omit the 'END OF DATA FILE' trailer."""
    text, enc = _read_text(pb01a)
    assert "END OF DATA" in text  # the bundled file has it
    truncated = text[: text.index("END OF DATA")].rstrip() + "\n"
    f = tmp_path / "no_marker.CSV"
    f.write_text(truncated, encoding=enc)

    result = read_from_csv(f, file_format="vanessen")
    assert isinstance(result, Dataset)
    assert len(result) == 2


def test_vanessen_single_letter_serial(tmp_path):
    """The default serial pattern should also match single-letter serials (e.g. W1619)."""
    text, enc = _read_text(pb01a)
    swapped = text.replace("AV319", "W1619").replace("PB01A", "PB16D")
    f = tmp_path / "single_letter.CSV"
    f.write_text(swapped, encoding=enc)

    ts = _as_timeseries(read_from_csv(f, file_format="vanessen"))
    assert ts.sensor == "W1619"
    assert ts.location == "PB16D"


def test_vanessen_explicit_location_sensor_override():
    """Explicit location/sensor kwargs should win over the regex extraction."""
    ts = _as_timeseries(
        read_from_csv(
            pb01a, file_format="vanessen", location="CUSTOMLOC", sensor="XYZ001"
        )
    )
    assert ts.location == "CUSTOMLOC"
    assert ts.sensor == "XYZ001"


def test_vanessen_uses_labelled_field_when_regex_fails(tmp_path):
    """A location/serial that doesn't match the regex is taken verbatim from the
    labelled header field ('Location' / 'Serial number') - no override needed."""
    text, enc = _read_text(pb01a)
    masked = text.replace("PB01A_moni_AV319", "neerijse").replace("AV319", "K5171")
    f = tmp_path / "neerijse.CSV"
    f.write_text(masked, encoding=enc)

    ts = _as_timeseries(read_from_csv(f, file_format="vanessen"))
    assert ts.location == "neerijse"
    assert ts.sensor == "K5171"


def test_vanessen_skipped_when_field_missing_then_override(tmp_path):
    """When the labelled 'Location' field is absent the file is skipped, but an
    explicit location= override still lets it parse."""
    text, enc = _read_text(pb01a)
    masked = "\n".join(ln for ln in text.splitlines() if "Location" not in ln) + "\n"
    f = tmp_path / "no_location.CSV"
    f.write_text(masked, encoding=enc)

    # No 'Location' field and the name isn't regex-matchable -> skipped.
    assert len(read_from_csv(f, file_format="vanessen")) == 0

    # Explicit override supplies the missing location.
    ts = _as_timeseries(read_from_csv(f, file_format="vanessen", location="neerijse"))
    assert ts.location == "neerijse"


def test_vanessen_semicolon_delimited(tmp_path):
    """Some exports use ';' as the column separator in the data block."""
    text, enc = _read_text(pb01a)
    lines = text.splitlines()
    out = []
    in_data = False
    for ln in lines:
        if ln.startswith("Date/time"):
            in_data = True
        if in_data and not ln.startswith("END"):
            ln = ln.replace(",", ";")
        out.append(ln)
    f = tmp_path / "semicolon.CSV"
    f.write_text("\n".join(out) + "\n", encoding=enc)

    result = read_from_csv(f, file_format="vanessen")
    assert len(result) == 2
