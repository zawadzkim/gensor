import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from gensor.core.timeseries import Timeseries

# =================== Test Timeseries methods ================================
dates = pd.date_range(start="2023-01-01", periods=10, freq="D", tz="UTC")
values = np.arange(1, 11)
ts = Timeseries(ts=pd.Series(values, index=dates), variable="temperature", unit="degc")


def test_resample():
    ts = Timeseries(
        ts=pd.Series(values, index=dates), variable="temperature", unit="degc"
    )

    resampled_ts = ts.resample("2D")

    assert len(resampled_ts.ts) == 5
    assert resampled_ts.ts.index.freq == "2D"


def test_difference():
    transformed_ts = ts.transform("difference")

    assert len(transformed_ts.ts) == len(ts.ts) - 1
    assert transformed_ts.transformation == "difference"


def test_log():
    transformed_ts = ts.transform("log")

    assert np.allclose(transformed_ts.ts.dropna(), np.log(values))
    assert transformed_ts.transformation == "log"


def test_square_root():
    transformed_ts = ts.transform("square_root")

    assert np.allclose(transformed_ts.ts.dropna(), np.sqrt(values))
    assert transformed_ts.transformation == "square_root"


def test_box_cox():
    transformed_ts = ts.transform("box_cox", lmbda=0)

    box_cox_values = stats.boxcox(values, lmbda=0)

    assert np.allclose(transformed_ts.ts.dropna(), box_cox_values)
    assert transformed_ts.transformation == "box-cox"


def test_standard_scaler():
    transformed_ts = ts.transform("standard_scaler")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    assert np.allclose(transformed_ts.ts, scaled_values)
    assert isinstance(transformed_ts.transformation, StandardScaler)


def test_minmax_scaler():
    transformed_ts = ts.transform("minmax_scaler")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    assert np.allclose(transformed_ts.ts, scaled_values)
    assert isinstance(transformed_ts.transformation, MinMaxScaler)


def test_robust_scaler():
    transformed_ts = ts.transform("robust_scaler")

    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    assert np.allclose(transformed_ts.ts, scaled_values)
    assert isinstance(transformed_ts.transformation, RobustScaler)


def test_maxabs_scaler():
    transformed_ts = ts.transform("maxabs_scaler")

    scaler = MaxAbsScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    assert np.allclose(transformed_ts.ts, scaled_values)
    assert isinstance(transformed_ts.transformation, MaxAbsScaler)


if __name__ == "__main__":
    pytest.main()
