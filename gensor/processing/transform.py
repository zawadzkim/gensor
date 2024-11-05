from typing import Any, Literal

import numpy as np
from pandas import Series
from scipy import stats
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class Transformation:
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

        Keword Arguments:
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

        Keyword Arguments:
            lmbda (float): The transformation parameter. If not provided, it is automatically estimated.

        Returns:
            pandas.Series: The Box-Cox transformed time series data.
        """
        lmbda = kwargs.get("lmbda")

        if (self.data <= 0).any():
            message = (
                "Box-Cox transformation requires all values to be strictly positive."
            )
            raise ValueError(message)

        # Box-Cox always returns a tuple: (transformed_data, lmbda)
        if lmbda is not None:
            transformed_data = stats.boxcox(self.data, lmbda=lmbda)
        else:
            transformed_data, lmbda = stats.boxcox(self.data, lmbda=lmbda)

        # Return the transformed series and mark the method used
        transformed_series = Series(transformed_data, index=self.data.index)
        return transformed_series, f"box-cox (lambda={lmbda})"

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
