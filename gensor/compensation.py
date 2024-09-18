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

from typing import Any

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
    drop_low_wc: bool = True

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

    def compensate(self, **kwargs: Any) -> Timeseries | None:
        """Perform compensation.

        Keyword Arguments:
            alignment_period (str): The alignment period for the timeseries.
                Default is 'H' (hourly).
            threshold_wc (float): The threshold for the absolute water column.
                Defaults to 0.5 m.

        Returns:
            Timeseries: A new Timeseries instance with the compensated data.
        """

        alignment_period = kwargs.get("alignment_period", "h")
        threshold_wc = kwargs.get("threshold_wc", 0.5)
        resample_params = {"freq": alignment_period, "agg_func": "mean"}
        resampled_ts = self.ts.resample(**resample_params)

        if isinstance(self.barometric, Timeseries):
            if self.ts == self.barometric:
                print("Skipping compensation: both timeseries are the same.")
                return None
            baro = self.barometric.resample(**resample_params).ts
        elif isinstance(self.barometric, float):
            baro = pd.Series(
                [self.barometric] * len(resampled_ts.ts), index=resampled_ts.ts.index
            )

        # dividing by 100 to convert water column from cmH2O to mH2O
        watercolumn_ts = resampled_ts.ts.sub(baro).divide(100).dropna()

        if self.drop_low_wc:
            watercolumn_ts_filtered = watercolumn_ts[
                watercolumn_ts.abs() > threshold_wc
            ]
            print(
                f"{len(watercolumn_ts) - len(watercolumn_ts_filtered)} records \
                    dropped due to low water column."
            )
            gwl = watercolumn_ts_filtered.add(float(resampled_ts.sensor_alt or 0))
        else:
            gwl = watercolumn_ts.add(float(resampled_ts.sensor_alt or 0))

        compensated = resampled_ts.model_copy(
            update={"ts": gwl, "unit": "m asl", "variable": "head"}
        )

        return compensated


def compensate(
    raw: Timeseries | Dataset,
    barometric: Timeseries | float,
    drop_low_wc: bool,
    **kwargs: Any,
) -> Timeseries | Dataset:
    """Constructor for the Comensate class object.

    Parameters:
        raw (Timeseries | Dataset): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
        drop_low_wc (bool): Whether to drop records where the absolute water column is
            less than or equal to the cutoff value. Defaults to True.
    """

    def _compensate_one(raw: Timeseries) -> Timeseries:
        comp = Compensator(ts=raw, barometric=barometric, drop_low_wc=drop_low_wc)
        return comp.compensate(**kwargs)

    if isinstance(raw, Timeseries):
        return _compensate_one(raw)

    elif isinstance(raw, Dataset):
        compensated_series = []
        for item in raw:
            compensated_series.append(_compensate_one(item))

        return raw.model_copy(update={"timeseries": compensated_series})
