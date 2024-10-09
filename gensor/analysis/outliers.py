import numba
import numpy as np
from pandas import Series
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


from typing import Any, Literal


class OutlierDetection:
    """Detecting outliers in groundwater timeseries data.

    Each method in this class returns a pandas.Series containing predicted outliers in
    the dataset.

    Methods:
        iqr: Use interquartile range (IQR).
        zscore: Use the z-score method.
        isolation_forest: Using the isolation forest algorithm.
        lof: Using the local outlier factor (LOF) method.
    """

    def __init__(
        self,
        data: Series,
        method: Literal["iqr", "zscore", "isolation_forest", "lof"],
        rolling: bool = False,
        window: int = 6,
        **kwargs: Any,
    ) -> None:
        """Find outliers in a time series using the specified method, with an option for rolling window."""

        FUNCS = {
            "iqr": self.iqr,
            "zscore": self.zscore,
            "isolation_forest": self.isolation_forest,
            "lof": self.lof,
        }

        method_func = FUNCS[method]

        if method in ["iqr", "zscore"]:
            y = (
                kwargs.get("k", 1.5)
                if method == "iqr"
                else kwargs.get("threshold", 3.0)
            )
            if rolling:
                roll = data.rolling(window=window)
                mask = roll.apply(lambda x: method_func(x, y), raw=True, engine="numba")
            else:
                mask = method_func(data.to_numpy(), y)

        bool_mask = mask.astype(bool)
        bool_mask_series = Series(bool_mask, index=data.index)
        self.outliers = data[bool_mask_series]

    @staticmethod
    @numba.njit(nogil=True)
    def iqr(data: np.ndarray, k: float) -> np.ndarray:
        """Use interquartile range (IQR).

        Parameters:
            data (pandas.Series): The time series data.

        Keyword Args:
            k (float): The multiplier for the IQR to define the range. Defaults to 1.5.

        Returns:
            np.ndarray: Binary mask representing the outliers as 1.
        """

        Q1 = np.percentile(data, 0.25)
        Q3 = np.percentile(data, 0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        return np.where((data < lower_bound) | (data > upper_bound), 1, 0)

    @staticmethod
    @numba.njit(nogil=True)
    def zscore(data: np.ndarray, threshold: float) -> np.ndarray:
        """Use the z-score method.

        Parameters:
            data (pandas.Series): The time series data.

        Keyword Args:
            threshold (float): The threshold for the z-score method. Defaults to 3.0.

        Returns:
            pandas.Series: Binary mask representing outliers.
        """

        mean = np.mean(data)
        std_dev = np.std(data)

        z_scores = np.abs((data - mean) / std_dev)
        return np.where(z_scores > threshold, 1, 0)

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