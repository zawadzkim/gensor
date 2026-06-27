from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pandas import Series
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Scale factor that makes the median absolute deviation a consistent estimator
# of the standard deviation for normally distributed data.
_MAD_TO_STD = 1.4826


class OutlierDetection:
    """Detecting outliers in groundwater timeseries data.

    Each method in this class returns a pandas.Series containing predicted outliers in
    the dataset.

    Methods:
        iqr: Use interquartile range (IQR).
        zscore: Use the z-score method.
        hampel: Use the Hampel filter (rolling median absolute deviation).
        isolation_forest: Using the isolation forest algorithm.
        lof: Using the local outlier factor (LOF) method.
    """

    def __init__(
        self,
        data: Series,
        method: Literal["iqr", "zscore", "hampel", "isolation_forest", "lof"],
        rolling: bool,
        window: int,
        **kwargs: Any,
    ) -> None:
        """Find outliers in a time series using the specified method, with an option for rolling window."""

        FUNCS: dict[str, Callable] = {
            "iqr": self.iqr,
            "zscore": self.zscore,
            "isolation_forest": self.isolation_forest,
            "lof": self.lof,
        }

        if method in ["iqr", "zscore"]:
            method_func = FUNCS[method]
            # For 'iqr' and 'zscore' methods
            y = (
                kwargs.get("k", 1.5)
                if method == "iqr"
                else kwargs.get("threshold", 3.0)
            )
            if rolling:
                roll = data.rolling(window=window)
                # `raw=True` hands each window to the detector as a plain ndarray
                # and requires a scalar return (0/1). Windows shorter than
                # `window` yield NaN; treat those as "not an outlier" so the
                # leading edge of the series is kept rather than dropped.
                mask = roll.apply(
                    lambda x: method_func(x, y, rolling=True), raw=True
                ).fillna(0)
            else:
                mask = method_func(data.to_numpy(), y, rolling=False)

            bool_mask = np.asarray(mask).astype(bool)
            bool_mask_series = Series(bool_mask, index=data.index)
            self.outliers = data[bool_mask_series]

        elif method == "hampel":
            self.outliers = self.hampel(data, window=window, **kwargs)

        else:
            # For 'isolation_forest' and 'lof' methods
            self.outliers = FUNCS[method](data, **kwargs)

    @staticmethod
    def iqr(data: np.ndarray, k: float, rolling: bool) -> Any:
        """Use interquartile range (IQR).

        Parameters:
            data (np.ndarray): The time series data (a window when ``rolling``).

        Keyword Args:
            k (float): The multiplier for the IQR to define the range. Defaults to 1.5.

        Returns:
            When ``rolling`` a scalar flag (1.0 outlier / 0.0 inlier) for the most
            recent point in the window; otherwise a binary mask marking outliers as 1.
        """

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        if rolling:
            return 1.0 if (data[-1] < lower_bound or data[-1] > upper_bound) else 0.0

        return np.where((data < lower_bound) | (data > upper_bound), 1, 0)

    @staticmethod
    def zscore(data: np.ndarray, threshold: float, rolling: bool) -> Any:
        """Use the z-score method.

        Parameters:
            data (np.ndarray): The time series data (a window when ``rolling``).

        Keyword Args:
            threshold (float): The threshold for the z-score method. Defaults to 3.0.

        Returns:
            When ``rolling`` a scalar flag (1.0 outlier / 0.0 inlier) for the most
            recent point in the window; otherwise a binary mask marking outliers as 1.
        """

        mean = np.mean(data)
        std_dev = np.std(data)

        z_scores = np.abs((data - mean) / std_dev)

        if rolling:
            return 1.0 if z_scores[-1] > threshold else 0.0
        return np.where(z_scores > threshold, 1, 0)

    @staticmethod
    def hampel(data: Series, window: int, n_sigma: float = 3.0) -> Series:
        """Use the Hampel filter (rolling median absolute deviation).

        For each point a centred window of size ``window`` is taken; the point is
        flagged when its absolute deviation from the window median exceeds
        ``n_sigma`` robust standard deviations, estimated as ``1.4826 * MAD``.
        Being median/MAD based it is far less sensitive to the very spikes it is
        meant to catch than the mean/std z-score, which makes it a good default
        for isolated sensor spikes.

        Parameters:
            data (pandas.Series): The time series data.
            window (int): Size of the centred rolling window (in samples).

        Keyword Args:
            n_sigma (float): Number of robust standard deviations beyond which a
                point is considered an outlier. Defaults to 3.0.

        Returns:
            pandas.Series: The subset of ``data`` flagged as outliers.
        """

        rolling = data.rolling(window=window, center=True, min_periods=1)
        median = rolling.median()
        mad = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        threshold = n_sigma * _MAD_TO_STD * mad

        deviation = (data - median).abs()
        # A zero threshold means the window has no spread; only flag a point when
        # it actually deviates (deviation > 0), so flat stretches stay intact.
        outlier_mask = deviation > threshold

        return data[outlier_mask]

    def isolation_forest(self, data: Series, **kwargs: Any) -> Series:
        """Using the isolation forest algorithm.

        Parameters:
            data (pandas.Series): The time series data.

        Keyword Args:
            n_estimators (int): The number of base estimators in the ensemble. Defaults to 100.
            max_samples (int | 'auto' | float): The number of samples to draw from X to train each base estimator. Defaults to 'auto'.
            contamination (float): The proportion of outliers in the data. Defaults to 0.01.
            max_features (int | float): The number of features to draw from X to train each base estimator. Defaults to 1.0.
            bootstrap (bool): Whether to use bootstrapping when sampling the data. Defaults to False.
            n_jobs (int): The number of jobs to run in parallel. Defaults to 1.
            random_state (int | RandomState | None): The random state to use. Defaults to None.
            verbose (int): The verbosity level. Defaults to 0.
            warm_start (bool): Whether to reuse the solution of the previous call to fit and add more estimators to the ensemble. Defaults to False.

        Note:
            For details on kwargs see: sklearn.ensemble.IsolationForest.
        """

        X = data.to_numpy().reshape(-1, 1)

        clf = IsolationForest(**kwargs)
        clf.fit(X)

        is_outlier = clf.predict(X)
        outliers: Series = data[is_outlier == -1]

        return outliers

    def lof(self, data: Series, **kwargs: Any) -> Series:
        """Using the local outlier factor (LOF) method.

        Parameters:
            data (pandas.Series): The time series data.

        Keyword Args:
            n_neighbors (int): The number of neighbors to consider for each sample. Defaults to 20.
            algorithm (str): The algorithm to use. Either 'auto', 'ball_tree', 'kd_tree' or 'brute'. Defaults to 'auto'.
            leaf_size (int): The leaf size of the tree. Defaults to 30.
            metric (str): The distance metric to use. Defaults to 'minkowski'.
            p (int): The power parameter for the Minkowski metric. Defaults to 2.
            contamination (float): The proportion of outliers in the data. Defaults to 0.01.
            novelty (bool): Whether to consider the samples as normal or outliers. Defaults to False.
            n_jobs (int): The number of jobs to run in parallel. Defaults to 1.
        Note:
            For details on kwargs see: sklearn.neighbors.LocalOutlierFactor.
        """

        X = data.to_numpy().reshape(-1, 1)

        clf = LocalOutlierFactor(**kwargs)

        is_outlier = clf.fit_predict(X)
        outliers: Series = data[is_outlier == -1]

        return outliers
