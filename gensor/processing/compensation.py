"""Compensating the raw data from the absolute pressure transducer to the actual water
level using the barometric pressure data.

Because van Essen Instrument divers are non-vented pressure transducers, to obtain the
pressure resulting from the water column above the logger (i.e. the water level), the
barometric pressure must be subtracted from the raw pressure measurements. In the
first step the function aligns the two series to the same time step and then subtracts
the barometric pressure from the raw pressure measurements. For short time periods (when
for instance a slug test is performed) the barometric pressure can be provided as a
single float value.

Subsequently the function filters out all records where the water column is less than or
equal to the cutoff value, and - always, regardless of the cutoff - every record with a
negative water column. The water column above a submerged sensor is physically
non-negative, so the near-zero readings taken while the logger is out of the water (which
produce erroneous results and spikes in the plots) and any negative values (out-of-water
/ noise / barometric-alignment artefacts) are all erroneous. The comparison is signed,
not on the absolute value, so large negative spikes are dropped rather than kept. The
cutoff defaults to 25 mm (``threshold_wc=0.025``) and is always applied; lower it to keep
shallower columns, or set it to 0 to drop only negatives.

Functions:

    water_column: Barometrically compensate raw pressure to the water column above the
        sensor (the first step, without adding the sensor altitude).
    compensate: Full compensation of raw sensor pressure to groundwater head, using
        ``water_column`` and then adding the sensor altitude.
"""

from typing import Literal

import pandas as pd
import pydantic as pyd

from ..core.dataset import Dataset
from ..core.timeseries import Timeseries
from ..exceptions import (
    InvalidMeasurementTypeError,
    MissingInputError,
)


class Compensator(pyd.BaseModel):
    """Compensate raw sensor pressure measurement with barometric pressure.

    Attributes:
        ts (Timeseries): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
    """

    ts: Timeseries
    barometric: Timeseries | float

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

    def water_column(
        self,
        alignment_period: Literal["D", "ME", "SME", "MS", "YE", "YS", "h", "min", "s"],
        threshold_wc: float | None,
        fieldwork_dates: list | None,
    ) -> Timeseries | None:
        """Compute the barometrically compensated water column above the sensor.

        Aligns the raw and barometric series to ``alignment_period``, subtracts the
        barometric pressure, converts cmH2O to mH2O, masks fieldwork days, and drops the
        out-of-water records (see ``threshold_wc``). This is the first step of
        :meth:`compensate` and can be used on its own to obtain just the water column
        height (it does not require ``sensor_alt``).

        Parameters:
            alignment_period Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']: The alignment period for the timeseries.
                Default is 'h'. See pandas offset aliases for definitinos.
            threshold_wc (float | None): Lower cutoff (in m) for the water column.
                Records at or below it are dropped, along with all negative water columns
                (which are always dropped as physically impossible). ``None`` is treated
                as ``0`` (drop only negatives).
            fieldwork_dates (Optional[list]): List of dates when fieldwork was done. All
                measurement from a fieldwork day will be set to None.

        Returns:
            Timeseries: A new Timeseries of the water column height in metres (variable
                'water_column', unit 'm'); dropped out-of-water records are kept in
                ``.outliers``. ``None`` if the raw and barometric series are the same.
        """

        resample_params = {"freq": alignment_period, "agg_func": pd.Series.mean}
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
        watercolumn_ts = resampled_ts.ts.sub(resampled_baro).divide(100).dropna()

        if not isinstance(watercolumn_ts.index, pd.DatetimeIndex):
            watercolumn_ts.index = pd.to_datetime(watercolumn_ts.index)

        if fieldwork_dates:
            fieldwork_timestamps = pd.to_datetime(fieldwork_dates).tz_localize(
                watercolumn_ts.index.tz
            )

            watercolumn_ts.loc[
                watercolumn_ts.index.normalize().isin(fieldwork_timestamps)
            ] = None

        # The water column above a submerged sensor is physically non-negative, so any
        # negative value (logger out of the water / noise / barometric misalignment) is
        # always discarded - regardless of the cutoff. On top of that, the near-zero
        # out-of-water band at or below ``threshold_wc`` (25 mm by default) is removed.
        # This always runs; pass a smaller ``threshold_wc`` to keep shallower columns, or
        # ``0`` to drop only negatives. NaN values (e.g. fieldwork-masked days) are left
        # in place as gaps. A signed comparison is essential: ``.abs() > threshold`` would
        # wrongly retain large-magnitude negatives.
        cutoff = 0.0 if threshold_wc is None else float(threshold_wc)
        invalid = (watercolumn_ts < 0) | (watercolumn_ts <= cutoff)
        watercolumn_ts_filtered = watercolumn_ts[~invalid]
        dropped_outliers = watercolumn_ts[invalid]

        if len(dropped_outliers):
            print(
                f"{len(dropped_outliers)} records dropped "
                f"(negative or <= {cutoff} m water column / out of water)."
            )

        return resampled_ts.model_copy(
            update={
                "ts": watercolumn_ts_filtered,
                "outliers": dropped_outliers,
                "unit": "m",
                "variable": "water_column",
            },
            deep=True,
        )

    def compensate(
        self,
        alignment_period: Literal["D", "ME", "SME", "MS", "YE", "YS", "h", "min", "s"],
        threshold_wc: float | None,
        fieldwork_dates: list | None,
    ) -> Timeseries | None:
        """Perform full compensation to groundwater head (m asl).

        Computes the water column with :meth:`water_column`, then adds the sensor
        altitude (``sensor_alt``) to express it as head above the reference datum.

        Parameters:
            alignment_period Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']: The alignment period for the timeseries.
                Default is 'h'. See pandas offset aliases for definitinos.
            threshold_wc (float | None): Lower cutoff (in m) for the water column; see
                :meth:`water_column`.
            fieldwork_dates (Optional[list]): List of dates when fieldwork was done. All
                measurement from a fieldwork day will be set to None.

        Returns:
            Timeseries: A new Timeseries instance with the compensated data and updated unit and variable. Optionally removed outliers are included.
        """
        watercolumn = self.water_column(
            alignment_period=alignment_period,
            threshold_wc=threshold_wc,
            fieldwork_dates=fieldwork_dates,
        )
        if watercolumn is None:
            return None

        gwl = watercolumn.ts.add(float(watercolumn.sensor_alt or 0))

        return watercolumn.model_copy(
            update={"ts": gwl, "unit": "m asl", "variable": "head"},
            deep=True,
        )


def _apply(
    step: Literal["compensate", "water_column"],
    raw: Timeseries | Dataset,
    barometric: Timeseries | float,
    alignment_period: Literal["D", "ME", "SME", "MS", "YE", "YS", "h", "min", "s"],
    threshold_wc: float | None,
    fieldwork_dates: dict | None,
    interpolate_method: str | None,
) -> Timeseries | Dataset | None:
    """Run a Compensator step (``compensate`` or ``water_column``) over a Timeseries or
    every Timeseries in a Dataset, applying per-location fieldwork dates and optional
    interpolation. Shared by :func:`compensate` and :func:`water_column`."""
    if fieldwork_dates is None:
        fieldwork_dates = {}

    def _one(item: Timeseries, dates: list | None) -> Timeseries | None:
        comp = Compensator(ts=item, barometric=barometric)
        result = getattr(comp, step)(
            alignment_period=alignment_period,
            threshold_wc=threshold_wc,
            fieldwork_dates=dates,
        )
        if result is not None and interpolate_method:
            # .interpolate() on a Timeseries is wrapped to return a Timeseries from the
            # original pandas.Series.interpolate().
            return result.interpolate(method=interpolate_method)  # type: ignore[no-any-return]
        return result

    if isinstance(raw, Timeseries):
        return _one(raw, fieldwork_dates.get(raw.location))

    elif isinstance(raw, Dataset):
        series = [_one(item, fieldwork_dates.get(item.location)) for item in raw]
        return raw.model_copy(update={"timeseries": series}, deep=True)


def compensate(
    raw: Timeseries | Dataset,
    barometric: Timeseries | float,
    alignment_period: Literal[
        "D", "ME", "SME", "MS", "YE", "YS", "h", "min", "s"
    ] = "h",
    threshold_wc: float | None = 0.025,
    fieldwork_dates: dict | None = None,
    interpolate_method: str | None = None,
) -> Timeseries | Dataset | None:
    """Compensate raw sensor pressure to groundwater head (m asl).

    Computes the water column (see :func:`water_column`) and adds the sensor altitude.

    Parameters:
        raw (Timeseries | Dataset): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
        alignment_period (Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']): The alignment period for the timeseries.
            Default is 'h'. See pandas offset aliases for definitinos.
        threshold_wc (float | None): Lower cutoff (in m) for the water column; records at
            or below it are dropped. Defaults to 0.025 m (25 mm) and is always applied;
            lower it to keep shallower columns, or set 0 to drop only negatives. Negative
            water columns are always dropped regardless, being physically impossible.
        fieldwork_dates (Dict[str, list]): Dictionary of location name and a list of
            fieldwork days. All records on the fieldwork day are set to None.
        interpolate_method (str): String representing the interpolate method as in
            pd.Series.interpolate() method.

    Returns:
        Timeseries | Dataset | None: head (variable 'head', unit 'm asl').
    """
    return _apply(
        "compensate", raw, barometric, alignment_period, threshold_wc,
        fieldwork_dates, interpolate_method,
    )


def water_column(
    raw: Timeseries | Dataset,
    barometric: Timeseries | float,
    alignment_period: Literal[
        "D", "ME", "SME", "MS", "YE", "YS", "h", "min", "s"
    ] = "h",
    threshold_wc: float | None = 0.025,
    fieldwork_dates: dict | None = None,
    interpolate_method: str | None = None,
) -> Timeseries | Dataset | None:
    """Barometrically compensate raw sensor pressure to the water column above the sensor.

    This is the first step of :func:`compensate` exposed on its own: subtract the
    barometric pressure, convert to mH2O, mask fieldwork days, and drop out-of-water
    records (see ``threshold_wc``) - without adding the sensor altitude, so the result is
    the water column height in metres (variable 'water_column', unit 'm') rather than head.

    Parameters:
        raw (Timeseries | Dataset): Raw sensor timeseries
        barometric (Timeseries | float): Barometric pressure timeseries or a single
            float value. If a float value is provided, it is assumed to be in cmH2O.
        alignment_period (Literal['D', 'ME', 'SME', 'MS', 'YE', 'YS', 'h', 'min', 's']): The alignment period for the timeseries.
            Default is 'h'. See pandas offset aliases for definitinos.
        threshold_wc (float | None): Lower cutoff (in m) for the water column; records at
            or below it are dropped. Defaults to 0.025 m (25 mm) and is always applied;
            lower it to keep shallower columns, or set 0 to drop only negatives. Negative
            water columns are always dropped regardless, being physically impossible.
        fieldwork_dates (Dict[str, list]): Dictionary of location name and a list of
            fieldwork days. All records on the fieldwork day are set to None.
        interpolate_method (str): String representing the interpolate method as in
            pd.Series.interpolate() method.

    Returns:
        Timeseries | Dataset | None: the water column height (variable 'water_column',
            unit 'm').
    """
    return _apply(
        "water_column", raw, barometric, alignment_period, threshold_wc,
        fieldwork_dates, interpolate_method,
    )
