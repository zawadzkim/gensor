from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Generic

import pandas as pd
import pydantic as pyd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gensor.core.base import BaseTimeseries, T
from gensor.db import DatabaseConnection
from gensor.exceptions import IndexOutOfRangeError

logger = logging.getLogger(__name__)


class Dataset(pyd.BaseModel, Generic[T]):
    """Store and operate on a collection of Timeseries.

    Attributes:
        timeseries (list[Timeseries]): A list of Timeseries objects.
    """

    timeseries: list[T | None] = pyd.Field(default_factory=list)

    def __iter__(self) -> Any:
        """Allows to iterate directly over the dataset."""
        return iter(self.timeseries)

    def __len__(self) -> int:
        """Gives the number of timeseries in the Dataset."""
        return len(self.timeseries)

    def __repr__(self) -> str:
        return f"Dataset({len(self)})"

    def __getitem__(self, key: int | str | list | tuple) -> T | None | Dataset:
        """Retrieve Timeseries by integer index, location name, or (location,
        variable[, unit]) tuple.

        - ``dataset[0]`` returns the Timeseries at that position (a reference).
        - ``dataset["PB01A"]`` returns the Timeseries at that location, or a
          Dataset if the location has several timeseries (e.g. pressure and
          temperature). A list of names (``dataset[["PB01A", "PB02A"]]``) always
          returns a Dataset.
        - ``dataset["PB01A", "pressure"]`` (or ``["PB01A", "pressure", "cmh2o"]``)
          narrows by variable/unit, returning a single Timeseries when one matches.
          For full control use :meth:`filter` / :meth:`one`.

        !!! warning
            Integer indexing returns a reference to the timeseries. Location /
            tuple indexing returns copies (it delegates to ``.filter()``).

        Parameters:
            key (int | str | list | tuple): Position, location name, list of
                names, or a (location, variable[, unit]) tuple.

        Returns:
            Timeseries | Dataset: The matching timeseries or a dataset of them.

        Raises:
            IndexOutOfRangeError: If an integer index is out of range.
            KeyError: If no timeseries matches the given location(s)/filters.
        """
        if isinstance(key, tuple):
            location, variable, unit = (*key, None, None)[:3]
            result = self.filter(location=location, variable=variable, unit=unit)
            if isinstance(result, Dataset) and len(result) == 0:
                message = f"No timeseries found for {key!r}."
                raise KeyError(message)
            return result

        if isinstance(key, (str, list)):
            result = self.filter(location=key)
            if isinstance(result, Dataset) and len(result) == 0:
                message = f"No timeseries found for location(s) {key!r}."
                raise KeyError(message)
            return result

        try:
            return self.timeseries[key]
        except IndexError:
            raise IndexOutOfRangeError(key, len(self)) from None

    def __contains__(self, location: object) -> bool:
        """Return True if any timeseries in the dataset has the given location."""
        return any(
            ts is not None and ts.location == location for ts in self.timeseries
        )

    def get_locations(self) -> list:
        """List all unique locations in the dataset, preserving first-seen order."""
        locations: list = []
        for ts in self.timeseries:
            if ts is not None and ts.location not in locations:
                locations.append(ts.location)
        return locations

    @property
    def coverage(self) -> Coverage:
        """Coverage summary of the dataset.

        Renders as a per-timeseries table (records and time span per location /
        variable / sensor) and exposes :meth:`Coverage.plot` for a coverage timeline.

        Examples:
            >>> ds.coverage          # the table  # doctest: +SKIP
            >>> ds.coverage.plot()   # the timeline  # doctest: +SKIP
        """
        return Coverage(self)

    def diff(
        self,
        *others: Dataset,
        labels: list[str] | None = None,
        key: tuple[str, ...] = ("location", "variable"),
    ) -> CoverageDiff:
        """Compare this dataset's coverage with one or more others.

        Convenience wrapper over :func:`gensor.diff`. ``labels`` names this dataset
        and the others (default ``ds0``, ``ds1`` ...).

        Examples:
            >>> raw.diff(trimmed, labels=["raw", "trimmed"]).plot()  # doctest: +SKIP
        """
        datasets = [self, *others]
        if labels is None:
            labels = [f"ds{i}" for i in range(len(datasets))]
        return diff(dict(zip(labels, datasets, strict=True)), key=key)

    def one(self, **filters: Any) -> T:
        """Return exactly one matching Timeseries.

        A convenience over :meth:`filter` for when a single result is expected:
        it always returns a Timeseries (never a Dataset) and raises if zero or
        more than one timeseries match - avoiding the "is it a Timeseries or a
        Dataset?" ambiguity of :meth:`filter` / ``dataset[name]``.

        Parameters:
            **filters: Same keyword filters as :meth:`filter` (location,
                variable, unit, sensor, ...).

        Returns:
            Timeseries: The single matching timeseries.

        Raises:
            ValueError: If zero or more than one timeseries match the filters.
        """
        result = self.filter(**filters)
        if isinstance(result, BaseTimeseries):
            return result

        count = len(result)
        message = f"Expected exactly one timeseries matching {filters}, found {count}."
        raise ValueError(message)

    def add(self, other: T | list[T] | Dataset) -> Dataset:
        """Appends new Timeseries to the Dataset.

        If an equal Timeseries already exists, merge the new data into the existing
        Timeseries, dropping duplicate timestamps.

        Parameters:
            other (Timeseries): The Timeseries object to add.
        """

        # I need to check for BaseTimeseries instance in the add() method, but also
        # type hint VarType T.
        if isinstance(other, list | Dataset):
            for ts in other:
                if isinstance(ts, BaseTimeseries):
                    self._add_single_timeseries(ts)  # type: ignore[arg-type]

        elif isinstance(other, BaseTimeseries):
            self._add_single_timeseries(other)

        return self

    def _add_single_timeseries(self, ts: T) -> None:
        """Adds a single Timeseries to the Dataset or merges if an equal one exists."""
        for i, existing_ts in enumerate(self.timeseries):
            if existing_ts == ts:
                self.timeseries[i] = existing_ts.concatenate(ts)
                return

        self.timeseries.append(ts)

        return

    def filter(
        self,
        location: str | list | None = None,
        variable: str | list | None = None,
        unit: str | list | None = None,
        exclude: dict[str, str | list] | None = None,
        **kwargs: str | list,
    ) -> T | Dataset:
        """Return a Timeseries or a new Dataset filtered by station, sensor,
        and/or variable.

        Any of ``location``/``variable``/``unit`` (and the keyword attributes) may be
        a single value or a list of values, matching a timeseries when its attribute
        equals (or is in) the given value(s).

        To filter by the *opposite* - dropping timeseries rather than selecting them -
        pass ``exclude``, a dict of ``{attribute: value | list}``. A timeseries is
        removed when it matches **all** conditions in ``exclude`` (e.g.
        ``exclude={"location": "PB16D"}`` drops that location, while
        ``exclude={"location": "PB03B", "sensor": "AV319"}`` drops only that one
        sensor at that location). ``exclude`` is applied after the include filters.

        Parameters:
            location (str | list, optional): The location name(s).
            variable (str | list, optional): The variable(s) being measured.
            unit (str | list, optional): Unit(s) of the measurement.
            exclude (dict, optional): ``{attribute: value | list}`` conditions whose
                (combined) match removes a timeseries from the result.
            **kwargs (str | list): Attributes of subclassed timeseries used for
                filtering (e.g., sensor, method).

        Returns:
            Timeseries | Dataset: A single Timeseries if exactly one match is found,
                                   or a new Dataset if multiple matches are found.
        """

        def as_list(value: str | list | None) -> list | None:
            return [value] if isinstance(value, str) else value

        def matches(ts: T, attr: str, value: list) -> bool:
            """Check the Timeseries has the attribute and its value is in ``value``."""
            if not hasattr(ts, attr):
                message = f"'{ts.__class__.__name__}' object has no attribute '{attr}'"
                raise AttributeError(message)
            return getattr(ts, attr) in value

        location, variable, unit = as_list(location), as_list(variable), as_list(unit)
        kwargs = {attr: as_list(value) for attr, value in kwargs.items()}
        exclude = {attr: as_list(value) for attr, value in (exclude or {}).items()}

        def keep(ts: T | None) -> bool:
            if ts is None:
                return False
            if location is not None and ts.location not in location:
                return False
            if variable is not None and ts.variable not in variable:
                return False
            if unit is not None and ts.unit not in unit:
                return False
            if not all(matches(ts, attr, value) for attr, value in kwargs.items()):
                return False
            if exclude and all(matches(ts, attr, value) for attr, value in exclude.items()):
                return False
            return True

        matching_timeseries = [ts for ts in self.timeseries if keep(ts)]

        if not matching_timeseries:
            return Dataset()

        if len(matching_timeseries) == 1:
            return matching_timeseries[0].model_copy(deep=True)

        return self.model_copy(update={"timeseries": matching_timeseries})

    def to_sql(self, db: DatabaseConnection) -> None:
        """Save the entire timeseries to a SQLite database.

        Parameters:
            db (DatabaseConnection): SQLite database connection object.
        """
        for ts in self.timeseries:
            if ts is None:
                continue
            if len(ts.ts) == 0:
                logger.info(
                    f"Skipping empty timeseries (location={ts.location!r}) - "
                    "nothing to write to the database."
                )
                continue
            ts.to_sql(db)
        return

    def plot(
        self,
        include_outliers: bool = False,
        plot_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the timeseries data, grouping by variable type.

        Parameters:
            include_outliers (bool): Whether to include outliers in the plot.
            plot_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.plot() method to customize the plot.
            legend_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.legend() to customize the legend.

        Returns:
            (fig, ax): Matplotlib figure and axes to allow further customization.
        """

        grouped_ts = defaultdict(list)

        for ts in self.timeseries:
            if ts:
                grouped_ts[ts.variable].append(ts)

        num_variables = len(grouped_ts)

        fig, axes = plt.subplots(
            num_variables, 1, figsize=(10, 5 * num_variables), sharex=True
        )

        if num_variables == 1:
            axes = [axes]

        for ax, (variable, ts_list) in zip(axes, grouped_ts.items(), strict=False):
            for ts in ts_list:
                ts.plot(
                    include_outliers=include_outliers,
                    ax=ax,
                    plot_kwargs=plot_kwargs,
                    legend_kwargs=legend_kwargs,
                )

            ax.set_title(f"Timeseries for {variable.capitalize()}")
            ax.set_xlabel("Time")

        fig.tight_layout()
        return fig, axes


def _coverage_segments(index: pd.DatetimeIndex, threshold: pd.Timedelta) -> list[tuple]:
    """Split a DatetimeIndex into ``(start, width)`` segments (in Matplotlib date
    units) of contiguous data, breaking wherever the gap between consecutive samples
    exceeds ``threshold``. Used to draw coverage bars with within-record gaps shown.
    """
    from matplotlib.dates import date2num

    index = index.sort_values()
    bars: list[tuple] = []
    seg_start = previous = index[0]
    for stamp in index[1:]:
        if stamp - previous > threshold:
            bars.append((date2num(seg_start), date2num(previous) - date2num(seg_start)))
            seg_start = stamp
        previous = stamp
    bars.append((date2num(seg_start), date2num(previous) - date2num(seg_start)))
    return bars


class Coverage:
    """Coverage summary of a :class:`Dataset`, returned by ``Dataset.coverage``.

    Holds a per-timeseries ``table`` (one row per location / variable / sensor with
    its record count and time span) and renders as that table in a notebook. Call
    :meth:`plot` for a coverage timeline (one row per location; bars span contiguous
    data, breaks mark gaps longer than ``max_gap``).
    """

    columns = ["location", "variable", "sensor", "unit", "records", "start", "end", "duration"]

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        table = pd.DataFrame(
            [
                {
                    "location": ts.location,
                    "variable": ts.variable,
                    "sensor": getattr(ts, "sensor", None),
                    "unit": ts.unit,
                    "records": len(ts.ts),
                    "start": ts.ts.index.min(),
                    "end": ts.ts.index.max(),
                    "duration": ts.ts.index.max() - ts.ts.index.min(),
                }
                for ts in dataset
                if ts is not None and len(ts.ts) > 0
            ],
            columns=self.columns,
        )
        if not table.empty:
            table = table.sort_values(["location", "variable"]).reset_index(drop=True)
        self.table = table

    def __repr__(self) -> str:
        return self.table.to_string(index=False)

    def _repr_html_(self) -> str:
        return self.table.to_html(index=False)

    def plot(
        self,
        max_gap: str = "7D",
        ax: Axes | None = None,
        color: str = "#1f4e79",
    ) -> tuple[Figure, Axes]:
        """Plot a coverage timeline: one row per location, with bars spanning
        contiguous data and breaks wherever the gap between consecutive samples
        exceeds ``max_gap``.

        Parameters:
            max_gap (str): pandas timedelta string; a gap longer than this splits a
                bar so within-record holes (e.g. a missing season) stay visible.
            ax (Axes | None): existing axes to draw on; a new figure is created if None.
            color (str): bar colour.

        Returns:
            (fig, ax): Matplotlib figure and axes.
        """
        threshold = pd.Timedelta(max_gap)
        locations = self._dataset.get_locations()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 0.35 * len(locations) + 1))
        else:
            fig = ax.figure

        for row, location in enumerate(locations):
            index = None
            for ts in self._dataset:
                if ts is None or ts.location != location or len(ts.ts) == 0:
                    continue
                index = ts.ts.index if index is None else index.union(ts.ts.index)
            if index is None or len(index) == 0:
                continue
            ax.broken_barh(_coverage_segments(index, threshold), (row - 0.4, 0.8), facecolors=color)

        ax.set_yticks(range(len(locations)))
        ax.set_yticklabels(locations, fontsize=8)
        ax.invert_yaxis()
        ax.xaxis_date()
        ax.set_title("Data coverage")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig, ax


class CoverageDiff:
    """Coverage comparison of two or more datasets, returned by :func:`gensor.diff`
    (or ``Dataset.diff``).

    Series are aligned across datasets by ``key`` (default ``("location",
    "variable")``); multiple sensors sharing a key are unioned and the sensor(s)
    reported. Renders as a wide ``table`` (per-dataset record count / start / end,
    plus ``present`` and ``status`` summary columns) and exposes :meth:`plot` for an
    N-way coverage timeline grouped by timeseries.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        key: tuple[str, ...] = ("location", "variable"),
    ) -> None:
        if len(datasets) < 2:
            message = "CoverageDiff needs at least two datasets to compare."
            raise ValueError(message)

        self._datasets = dict(datasets)
        self.key = tuple(key)
        self.labels = list(datasets)

        # per label: key-tuple -> {sensor, records, start, end, index}
        self._coverage: dict[str, dict[tuple, dict]] = {}
        for label, dataset in datasets.items():
            grouped: dict[tuple, dict] = {}
            for ts in dataset:
                if ts is None or len(ts.ts) == 0:
                    continue
                k = tuple(getattr(ts, attr) for attr in self.key)
                entry = grouped.setdefault(k, {"sensors": set(), "index": None})
                entry["index"] = (
                    ts.ts.index
                    if entry["index"] is None
                    else entry["index"].union(ts.ts.index)
                )
                entry["sensors"].add(getattr(ts, "sensor", None))
            self._coverage[label] = {
                k: {
                    "sensor": "+".join(sorted(s for s in v["sensors"] if s)) or None,
                    "records": len(v["index"]),
                    "start": v["index"].min(),
                    "end": v["index"].max(),
                    "index": v["index"].sort_values(),
                }
                for k, v in grouped.items()
            }

        self.keys = sorted({k for cov in self._coverage.values() for k in cov})
        self.table = self._build_table()

    def _status(self, k: tuple) -> str:
        present = [lab for lab in self.labels if k in self._coverage[lab]]
        if len(present) < len(self.labels):
            return "only " + ", ".join(present)
        records = {self._coverage[lab][k]["records"] for lab in self.labels}
        spans = {
            (self._coverage[lab][k]["start"], self._coverage[lab][k]["end"])
            for lab in self.labels
        }
        if len(records) == 1 and len(spans) == 1:
            return "identical"
        return "span differs" if len(spans) > 1 else "records differ"

    def _build_table(self) -> pd.DataFrame:
        defaults = {"sensor": None, "records": 0, "start": pd.NaT, "end": pd.NaT}
        data: dict[tuple, list] = {}
        for label in self.labels:
            cov = self._coverage[label]
            for metric in ("sensor", "records", "start", "end"):
                data[(label, metric)] = [
                    cov.get(k, defaults).get(metric, defaults[metric]) for k in self.keys
                ]
        data[("summary", "present")] = [
            sum(k in self._coverage[lab] for lab in self.labels) for k in self.keys
        ]
        data[("summary", "status")] = [self._status(k) for k in self.keys]

        table = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(self.keys, names=self.key),
        )
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        return table

    def __repr__(self) -> str:
        return self.table.to_string()

    def _repr_html_(self) -> str:
        return self.table.to_html()

    def plot(
        self,
        max_gap: str = "7D",
        ax: Axes | None = None,
        colors: dict[str, Any] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot an N-way coverage timeline grouped by timeseries.

        One row per ``key`` (e.g. location + variable); within each row a coverage
        sub-bar per dataset (colour-coded, with a legend). Series present in only one
        dataset, or covering different spans, are immediately visible.

        Parameters:
            max_gap (str): pandas timedelta string; gaps longer than this split a bar.
            ax (Axes | None): existing axes to draw on; a new figure is created if None.
            colors (dict | None): optional ``{label: colour}`` mapping.

        Returns:
            (fig, ax): Matplotlib figure and axes.
        """
        from matplotlib.patches import Patch

        threshold = pd.Timedelta(max_gap)
        if colors is None:
            cmap = plt.get_cmap("tab10")
            colors = {lab: cmap(i % 10) for i, lab in enumerate(self.labels)}

        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 0.45 * len(self.keys) + 1.5))
        else:
            fig = ax.figure

        n = len(self.labels)
        sub_h = 0.8 / n
        for row, k in enumerate(self.keys):
            for j, label in enumerate(self.labels):
                info = self._coverage[label].get(k)
                if info is None:
                    continue
                y = row - 0.4 + j * sub_h
                ax.broken_barh(
                    _coverage_segments(info["index"], threshold),
                    (y, sub_h * 0.9),
                    facecolors=colors[label],
                )

        ax.set_yticks(range(len(self.keys)))
        ax.set_yticklabels([" ".join(map(str, k)) for k in self.keys], fontsize=7)
        ax.invert_yaxis()
        ax.xaxis_date()
        ax.set_title("Coverage diff")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(
            handles=[Patch(facecolor=colors[lab], label=lab) for lab in self.labels],
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=True,
        )
        fig.tight_layout()
        return fig, ax


def diff(
    datasets: dict[str, Dataset] | list[Dataset],
    key: tuple[str, ...] = ("location", "variable"),
) -> CoverageDiff:
    """Compare the coverage of two or more datasets.

    Parameters:
        datasets: a mapping ``{label: Dataset}`` (preferred - labels name the columns
            and legend) or a list of datasets (auto-labelled ``ds0``, ``ds1`` ...).
        key: attributes used to align series across datasets (default
            ``("location", "variable")``).

    Returns:
        CoverageDiff: renders as a comparison table; ``.plot()`` draws the timeline.
    """
    if isinstance(datasets, Dataset):
        message = "Pass two or more datasets to diff(), e.g. diff({'a': ds1, 'b': ds2})."
        raise TypeError(message)
    if not isinstance(datasets, dict):
        datasets = {f"ds{i}": d for i, d in enumerate(datasets)}
    return CoverageDiff(datasets, key=key)
