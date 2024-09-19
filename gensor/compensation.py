"""Compensating the raw data from the absolute pressure transducer to the actual water
level using the barometric pressure data.

Because van Essen Instrument divers are non-vented pressure transducers, to obtain the
pressure resulting from the water column above the logger (i.e. the water level), the
barometric pressure must be subtracted from the raw pressure measurements. In the
first step the function aligns the two series to the same time step and then subtracts
the barometric pressure from the raw pressure measurements. For short time periods (when
for instance a slug test is performed) the barometric pressure can be provided as a
single float value.

Subsequently the function filters out all records where the absolute water column is
less than or equal to the cutoff value. This is because when the logger is out of the
water when the measurement is taken, the absolute water column is close to zero,
producing erroneous results and spikes in the plots. The cutoff value is set to 5 cm by
default, but can be adjusted using the cutoff_wc kwarg.

Functions:

    compensate: Compensate raw sensor pressure measurement with barometric pressure.
"""

from typing import Literal

import pandas as pd
import pydantic as pyd

from .dtypes import Dataset, Timeseries
from .exceptions import (
    InvalidMeasurementTypeError,
    MissingInputError,
)


class Compensator(pyd.BaseModel):
    """Compensate raw sensor pressure measurement with barometric pressure.

    Attributes:
        ts (Timeseries): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
        drop_low_wc (bool): Whether to drop records where the absolute water column is
            less than or equal to the cutoff value. Defaults to True.

    """

    ts: Timeseries
    barometric: Timeseries | float
    drop_low_wc: bool

    @pyd.field_validator("ts", "barometric", mode="before")
    def validate_timeseries_type(cls, v: Timeseries) -> Timeseries:
        if isinstance(v, Timeseries) and v.variable != "pressure":
            raise InvalidMeasurementTypeError()
        return v

    @pyd.field_validator("ts")
    def validate_sensor_information(cls, v: Timeseries) -> Timeseries:
        if v.sensor is not None and not v.sensor_alt:
            raise MissingInputError("sensor_alt")
        return v

    def compensate(self,
                   alignment_period: Literal['D', 'ME', 'SME',
                                             'MS', 'YE', 'YS', 'h', 'min', 's'],
                   threshold_wc: float) -> Timeseries | None:
        """Perform compensation.

        Parameters:
            alignment_period Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']: The alignment period for the timeseries.
                Default is 'h'. See pandas offset aliases for definitinos.
            threshold_wc (float): The threshold for the absolute water column.
                Defaults to 0.5 m.

        Returns:
            Timeseries: A new Timeseries instance with the compensated data and updated unit and variable. Optionally removed outliers are included.
        """

        resample_params = {"freq": alignment_period, "agg_func": "mean"}
        resampled_ts = self.ts.resample(**resample_params)

        if isinstance(self.barometric, Timeseries):
            if self.ts == self.barometric:
                print("Skipping compensation: both timeseries are the same.")
                return None
            resampled_baro = self.barometric.resample(**resample_params).ts

        elif isinstance(self.barometric, float):
            resampled_baro = pd.Series(
                [self.barometric] * len(resampled_ts.ts), index=resampled_ts.ts.index
            )

        # dividing by 100 to convert water column from cmH2O to mH2O
        watercolumn_ts = resampled_ts.ts.sub(
            resampled_baro).divide(100).dropna()

        if self.drop_low_wc:
            watercolumn_ts_filtered = watercolumn_ts[
                watercolumn_ts.abs() > threshold_wc
            ]

            dropped_outliers = watercolumn_ts[
                watercolumn_ts.abs() <= threshold_wc
            ]

            print(
                f"{len(dropped_outliers)} records \
                    dropped due to low water column."
            )
            gwl = watercolumn_ts_filtered.add(
                float(resampled_ts.sensor_alt or 0))

            compensated = resampled_ts.model_copy(
                update={"ts": gwl,
                        "outliers": dropped_outliers,
                        "unit": "m asl",
                        "variable": "head"},
                deep=True
            )
        else:
            gwl = watercolumn_ts.add(float(resampled_ts.sensor_alt or 0))

            compensated = resampled_ts.model_copy(
                update={"ts": gwl, "unit": "m asl", "variable": "head"},
                deep=True
            )

        return compensated


def compensate(
    raw: Timeseries | Dataset,
    barometric: Timeseries | float,
    drop_low_wc: bool = False,
    alignment_period: Literal['D', 'ME', 'SME',
                              'MS', 'YE', 'YS', 'h', 'min', 's'] = 'h',
    threshold_wc: float = 0.5,
) -> Timeseries | Dataset:
    """Constructor for the Comensator object.

    Parameters:
        raw (Timeseries | Dataset): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
        drop_low_wc (bool): Whether to drop records where the absolute water column is
            less than or equal to the cutoff value. Defaults to True.
        alignment_period (Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']): The alignment period for the timeseries.
            Default is 'h'. See pandas offset aliases for definitinos.
        threshold_wc (float): The threshold for the absolute water column.
            Default is 0.5 m.
    """

    def _compensate_one(raw: Timeseries) -> Timeseries:
        comp = Compensator(ts=raw, barometric=barometric,
                           drop_low_wc=drop_low_wc)
        return comp.compensate(alignment_period=alignment_period,
                               threshold_wc=threshold_wc)

    if isinstance(raw, Timeseries):
        return _compensate_one(raw)

    elif isinstance(raw, Dataset):
        compensated_series = []
        for item in raw:
            compensated_series.append(_compensate_one(item))

        return raw.model_copy(update={"timeseries": compensated_series}, deep=True)
