"""Tests for gensor.analysis.outliers via Timeseries.detect_outliers."""

import numpy as np
import pandas as pd
import pytest

from gensor import Timeseries


def _make_ts(values: np.ndarray) -> Timeseries:
    index = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    series = pd.Series(values, index=index)
    return Timeseries(
        ts=series, variable="pressure", unit="cmh2o", location="test", sensor=None
    )


@pytest.fixture
def clean_with_spikes() -> tuple[Timeseries, list[int]]:
    """A flat, low-noise series with two large injected spikes."""
    rng = np.random.default_rng(42)
    values = rng.normal(100.0, 1.0, size=300)
    spike_positions = [50, 200]
    values[spike_positions[0]] = 500.0
    values[spike_positions[1]] = -300.0
    return _make_ts(values), spike_positions


@pytest.mark.parametrize("method", ["iqr", "zscore", "hampel"])
@pytest.mark.parametrize("rolling", [False, True])
def test_detect_outliers_runs_and_keeps_signal(method, rolling, clean_with_spikes):
    """Every method/mode runs without raising and keeps the bulk of the data.

    Regression test for the rolling iqr/zscore crash
    ("only 0-dimensional arrays can be converted to Python scalars") caused by
    the rolling detector returning a 1-element array instead of a scalar.
    """
    ts, _ = clean_with_spikes
    filtered = ts.detect_outliers(method=method, rolling=rolling, window=12)

    # The two spikes are removable; the vast majority of points must survive.
    assert not filtered.ts.empty
    assert len(filtered.ts) >= len(ts.ts) - 30


def test_iqr_uses_real_quartiles(clean_with_spikes):
    """Non-rolling IQR must use the 25th/75th percentiles, not 0.25/0.75.

    With the old `np.percentile(data, 0.25/0.75)` bug the bounds collapsed near
    the minimum and removed the overwhelming majority of clean points.
    """
    ts, spikes = clean_with_spikes
    filtered = ts.detect_outliers(method="iqr", rolling=False, window=12)

    # Only a handful of points should be removed, not hundreds.
    removed = len(ts.ts) - len(filtered.ts)
    assert removed <= 15
    # The injected spikes must be among the removed points.
    for pos in spikes:
        assert ts.ts.index[pos] not in filtered.ts.index


def test_hampel_flags_spikes_and_keeps_flat_regions(clean_with_spikes):
    ts, spikes = clean_with_spikes
    filtered = ts.detect_outliers(method="hampel", window=12, n_sigma=3.0)

    for pos in spikes:
        assert ts.ts.index[pos] not in filtered.ts.index
    # A near-flat, spike-free series should lose only the spikes.
    assert len(filtered.ts) >= len(ts.ts) - 10


def test_hampel_keeps_perfectly_flat_series():
    """A constant series has zero spread; nothing should be flagged."""
    ts = _make_ts(np.full(100, 42.0))
    filtered = ts.detect_outliers(method="hampel", window=11)
    assert len(filtered.ts) == len(ts.ts)


def test_rolling_does_not_drop_leading_window(clean_with_spikes):
    """The leading points (incomplete window -> NaN mask) must be retained."""
    ts, _ = clean_with_spikes
    filtered = ts.detect_outliers(method="zscore", rolling=True, window=12)
    # The very first timestamp is not an outlier and must survive.
    assert ts.ts.index[0] in filtered.ts.index


def test_detect_outliers_marks_without_removing(clean_with_spikes):
    ts, spikes = clean_with_spikes
    result = ts.detect_outliers(method="hampel", window=12, remove=False)
    # remove=False returns the original series but records the outliers.
    assert len(result.ts) == len(ts.ts)
    assert len(result.outliers) >= len(spikes)
