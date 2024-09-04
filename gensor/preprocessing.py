"""Class and methods for preprocessing groundwater level data."""

from typing import Any, Literal

import numpy as np
from pandas import Series
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class Transform:
    def __init__(
        self,
        data: Series,
        method: Literal[
            "difference",
            "log",
            "square_root",
            "box_cox",
            "standard_scaler",
            "minmax_scaler",
            "robust_scaler",
            "maxabs_scaler",
        ],
        **kwargs: Any,
    ) -> None:
        self.data = data

        if method == "difference":
            self.transformed_data, self.scaler = self.difference(**kwargs)
        elif method == "log":
            self.transformed_data, self.scaler = self.log()
        elif method == "square_root":
            self.transformed_data, self.scaler = self.square_root()
        elif method == "box_cox":
            self.transformed_data, self.scaler = self.box_cox(**kwargs)
        elif method == "standard_scaler":
            self.transformed_data, self.scaler = self.standard_scaler()
        elif method == "minmax_scaler":
            self.transformed_data, self.scaler = self.minmax_scaler()
        elif method == "robust_scaler":
            self.transformed_data, self.scaler = self.robust_scaler()
        elif method == "maxabs_scaler":
            self.transformed_data, self.scaler = self.maxabs_scaler()
        else:
            raise NotImplementedError()

    def get_transformation(self) -> tuple:
        return self.transformed_data, self.scaler

    def difference(self, **kwargs: int) -> tuple[Series, str]:
        """Difference the time series data.

        Args:
            periods (int): The number of periods to shift. Defaults to 1.

        Returns:
            pandas.Series: The differenced time series data.
        """
        periods = kwargs.get("periods", 1)
        transformed = self.data.diff(periods=periods).dropna()
        return (transformed, "difference")

    def log(self) -> tuple[Series, str]:
        """Take the natural logarithm of the time series data.

        Returns:
            pandas.Series: The natural logarithm of the time series data.
        """
        transformed = self.data.apply(lambda x: x if x <= 0 else np.log(x))
        return (transformed, "log")

    def square_root(self) -> tuple[Series, str]:
        """Take the square root of the time series data.

        Returns:
            pandas.Series: The square root of the time series data.
        """
        transformed = self.data.apply(lambda x: x if x <= 0 else np.sqrt(x))
        return (transformed, "square_root")

    def box_cox(self, **kwargs: float) -> tuple[Series, str]:
        """Apply the Box-Cox transformation to the time series data. Only works
            for all positive datasets!

        Args:
            lmbda (float): The transformation parameter. Defaults to 0.

        Returns:
            pandas.Series: The Box-Cox transformed time series data.
        """

        lmbda = kwargs.get("lmbda", None)

        if (self.data <= 0).any():
            message = (
                "Box-Cox transformation requires all values to be strictly positive."
            )
            raise ValueError(message)

        if not lmbda:
            result = stats.boxcox(self.data, lmbda=lmbda)
            transformed_series = Series(result, index=self.data.index)
        else:
            result = stats.boxcox(self.data, lmbda=lmbda)
            transformed_series = Series(result[0], index=self.data.index)

        return transformed_series, "box-cox"

    def standard_scaler(self) -> tuple[Series, Any]:
        """Normalize a pandas Series using StandardScaler."""
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(
            self.data.to_numpy().reshape(-1, 1)
        ).flatten()
        scaled_series = Series(scaled_values, index=self.data.index)
        return scaled_series, scaler

    def minmax_scaler(self) -> tuple[Series, Any]:
        """Normalize a pandas Series using MinMaxScaler."""
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(
            self.data.to_numpy().reshape(-1, 1)
        ).flatten()
        scaled_series = Series(scaled_values, index=self.data.index)
        return scaled_series, scaler

    def robust_scaler(self) -> tuple[Series, Any]:
        """Normalize a pandas Series using RobustScaler."""
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(
            self.data.to_numpy().reshape(-1, 1)
        ).flatten()
        scaled_series = Series(scaled_values, index=self.data.index)
        return scaled_series, scaler

    def maxabs_scaler(self) -> tuple[Series, Any]:
        """Normalize a pandas Series using MaxAbsScaler."""
        scaler = MaxAbsScaler()
        scaled_values = scaler.fit_transform(
            self.data.to_numpy().reshape(-1, 1)
        ).flatten()
        scaled_series = Series(scaled_values, index=self.data.index)
        return scaled_series, scaler


class OutlierDetection:
    """Class for detecting outliers in time series data."""

    def __init__(
        self,
        data: Series,
        method: Literal["iqr", "zscore", "isolation_forest", "lof"],
        **kwargs: Any,
    ) -> None:
        """Find outliers in a time series using the specified method."""
        if method == "iqr":
            self.outliers = self.iqr(data, **kwargs)
        elif method == "zscore":
            self.outliers = self.zscore(data, **kwargs)
        elif method == "isolation_forest":
            self.outliers = self.isolation_forest(data, **kwargs)
        elif method == "lof":
            self.outliers = self.lof(data, **kwargs)
        else:
            raise NotImplementedError()

    def iqr(self, data: Series, **kwargs: float) -> Series:
        """Use interquartile range (IQR).

        Parameters:
            data (pandas.Series): The time series data.

        Keyword Args:
            k (float): The multiplier for the IQR to define the range. Defaults to 1.5.

        Returns:
            pandas.Series: Outliers detected in the data.
        """

        k: float = kwargs.get("k", 1.5)

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return outliers

    def zscore(self, data: Series, **kwargs: float) -> Series:
        """Detect outliers in a time series using the z-score method.

        Args:
            data (pandas.Series): The time series data.

        Keyword Args:
            threshold (float): The threshold for the z-score method. Defaults to 3.0.

        Returns:
            pandas.Series: Outliers detected in the data.
        """

        threshold = kwargs.get("threshold", 3.0)

        mean = data.mean()
        std_dev = data.std()

        outliers: Series = data[(data - mean).abs() > threshold * std_dev]

        return outliers

    def isolation_forest(self, data: Series, **kwargs: Any) -> Series:
        """Detect outliers in a time series using the isolation forest method.

        Args:
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
        """Detect outliers in a time series using the local outlier factor (LOF) method.

        Args:
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
