from __future__ import annotations

import logging
from typing import Any, ClassVar, Generic

import pandas as pd
import pydantic as pyd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gensor.core.base import BaseTimeseries, T
from gensor.db import DatabaseConnection
from gensor.exceptions import IndexOutOfRangeError

logger = logging.getLogger(__name__)


def _split(value: str | list | None) -> tuple[set, set]:
    """Split an attribute spec into (include, exclude) value sets.

    A leading ``~`` on a string value moves it to the exclude set; ``None`` (or an
    empty spec) constrains nothing. Shared by ``Dataset.filter`` and ``Where``.
    """
    if value is None:
        return set(), set()
    values = [value] if isinstance(value, str) else value
    include: set = set()
    exclude: set = set()
    for v in values:
        (exclude if v.startswith("~") else include).add(
            v[1:] if v.startswith("~") else v
        )
    return include, exclude


class Where:
    """A composable predicate over a Timeseries' attributes, for ``Dataset.filter``/``drop``.

    A leaf ``Where(**conditions)`` matches a Timeseries when **every** condition holds;
    each condition matches when the timeseries' attribute equals (or is in, for a list)
    the given value(s), and a leading ``~`` on a value negates that single condition.
    Compose leaves with ``&`` (and), ``|`` (or) and ``~`` (not) to express anything the
    per-attribute keyword filters can't - in particular a *combined* exclusion::

        ~Where(location="PB03B", sensor="AV319")            # not (PB03B and AV319)
        Where(variable="pressure") & ~Where(location="PB16D")
        Where(location="PB16A") | Where(location="PB16B")

    Pass instances straight to ``Dataset.filter`` (keep matches) or ``Dataset.drop``
    (remove matches); they are AND-ed with the keyword filters in the same call.
    """

    def __init__(self, _test: Any = None, **conditions: str | list) -> None:
        self._conditions = conditions
        self._test = _test if _test is not None else self._compile(conditions)

    @staticmethod
    def _compile(conditions: dict) -> Any:
        specs = {attr: _split(value) for attr, value in conditions.items()}

        def test(ts: Any) -> bool:
            for attr, (include, exclude) in specs.items():
                if not hasattr(ts, attr):
                    message = (
                        f"'{ts.__class__.__name__}' object has no attribute '{attr}'"
                    )
                    raise AttributeError(message)
                actual = getattr(ts, attr)
                if (include and actual not in include) or actual in exclude:
                    return False
            return True

        return test

    def __call__(self, ts: Any) -> bool:
        return bool(self._test(ts))

    def __invert__(self) -> Where:
        return Where(_test=lambda ts: not self._test(ts))

    def __and__(self, other: Where) -> Where:
        return Where(_test=lambda ts: self._test(ts) and other(ts))

    def __or__(self, other: Where) -> Where:
        return Where(_test=lambda ts: self._test(ts) or other(ts))

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._conditions.items())
        return f"Where({body})"


class DatasetIndexer:
    """Applies a pandas ``.loc`` selection to every Timeseries in a Dataset.

    Returned by :attr:`Dataset.loc`. ``ds.loc[start:end]`` slices each timeseries by label
    (e.g. a date range) via its own ``.loc`` and returns a new Dataset of the results.
    Intended for label slices; a key that selects a single scalar from a timeseries (a
    point lookup) is rejected, since the per-series scalars can't form a Dataset.
    """

    def __init__(self, parent: Dataset) -> None:
        self.parent = parent

    def __getitem__(self, key: Any) -> Dataset:
        sliced: list = []
        for ts in self.parent.timeseries:
            if ts is None:
                sliced.append(None)
                continue
            result = ts.loc[key]
            if not isinstance(result, BaseTimeseries):
                message = (
                    "Dataset.loc expects a label slice (e.g. ds.loc[start:end]); "
                    f"key {key!r} selected a scalar from a timeseries."
                )
                raise TypeError(message)
            sliced.append(result)
        return self.parent.model_copy(update={"timeseries": sliced}, deep=False)


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

        if isinstance(key, str | list):
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
        return any(ts is not None and ts.location == location for ts in self.timeseries)

    def get_locations(self) -> list:
        """List all unique locations in the dataset, preserving first-seen order."""
        locations: list = []
        for ts in self.timeseries:
            if ts is not None and ts.location not in locations:
                locations.append(ts.location)
        return locations

    @property
    def loc(self) -> DatasetIndexer:
        """Label-based selection applied to every timeseries in the dataset.

        ``ds.loc[start:end]`` returns a new Dataset where each timeseries is sliced by
        ``.loc[start:end]`` (e.g. a date range), forwarding the key to each series' own
        pandas ``.loc``. Empty slices yield empty timeseries (every series is kept).

        Examples:
            >>> ds.loc["2021-01-01":"2021-12-31"]  # doctest: +SKIP
        """
        return DatasetIndexer(self)

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

    @property
    def info(self) -> pd.DataFrame:
        """Per-timeseries metadata summary, rendered as a table.

        One row per timeseries — ``location``, ``variable``, ``sensor``, the number of
        ``records``, and the ``start`` / ``end`` of its time span. A quick look at what
        a Dataset holds before processing it (the default repr only shows the timeseries
        count). See :attr:`coverage` for a plottable version and :func:`gensor.diff` to
        line this up across datasets.

        Examples:
            >>> ds.info  # doctest: +SKIP
        """
        columns = ["location", "variable", "sensor", "records", "start", "end"]
        table = pd.DataFrame(
            [
                {
                    "location": ts.location,
                    "variable": ts.variable,
                    "sensor": getattr(ts, "sensor", None),
                    "records": len(ts.ts),
                    "start": ts.ts.index.min(),
                    "end": ts.ts.index.max(),
                }
                for ts in self.timeseries
                if ts is not None and len(ts.ts) > 0
            ],
            columns=columns,
        )
        if not table.empty:
            table = table.sort_values(["location", "variable", "sensor"]).reset_index(
                drop=True
            )
        return table

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
        *predicates: Where,
        location: str | list | None = None,
        variable: str | list | None = None,
        unit: str | list | None = None,
        **kwargs: str | list,
    ) -> T | Dataset:
        """Return a Timeseries or a new Dataset filtered by station, sensor,
        and/or variable.

        Any of ``location``/``variable``/``unit`` (and the keyword attributes) may be
        a single value or a list of values, matching a timeseries when its attribute
        equals (or is in) the given value(s).

        Prefix a value with ``~`` to *negate* it - drop timeseries with that value
        rather than keep them (e.g. ``location="~PB16D"`` keeps everything except
        PB16D; ``sensor="~AV319"`` drops just that sensor). Positive and negated
        values may be mixed within one attribute and across attributes; for a given
        attribute a timeseries is kept when its value is in the positives (if any are
        given) **and** not in the negatives, and attributes are AND-ed together.

        For conditions the per-attribute keywords can't express - notably a *combined*
        match across attributes - pass one or more :class:`Where` predicates
        positionally. ``filter(~Where(location="PB03B", sensor="AV319"))`` drops only that
        sensor at that location (the whole combination negated as a unit), while
        ``filter(Where(location="PB16A") | Where(location="PB16B"))`` keeps either.
        Predicates are AND-ed with the keyword filters.

        Parameters:
            *predicates (Where): Predicate objects; all must match for a timeseries to
                be kept (combine with ``& | ~``).
            location (str | list, optional): The location name(s); ``~`` negates.
            variable (str | list, optional): The variable(s) being measured; ``~`` negates.
            unit (str | list, optional): Unit(s) of the measurement; ``~`` negates.
            **kwargs (str | list): Attributes of subclassed timeseries used for
                filtering (e.g., sensor, method); ``~`` negates.

        Returns:
            Timeseries | Dataset: A single Timeseries if exactly one match is found,
                                   or a new Dataset if multiple matches are found.
        """
        keep = self._matcher(predicates, location, variable, unit, kwargs)
        matching_timeseries = [ts for ts in self.timeseries if keep(ts)]

        if not matching_timeseries:
            return Dataset()

        if len(matching_timeseries) == 1:
            return matching_timeseries[0].model_copy(deep=True)

        return self.model_copy(update={"timeseries": matching_timeseries})

    def pop(
        self,
        *predicates: Where,
        location: str | list | None = None,
        variable: str | list | None = None,
        unit: str | list | None = None,
        **kwargs: str | list,
    ) -> T | Dataset:
        """Remove and return the matching timeseries, mutating the Dataset in place.

        Selection works exactly like :meth:`filter` (same ``location`` / ``variable`` /
        ``unit`` / keyword filters, ``~`` negation, and :class:`Where` predicates), but
        the matched timeseries are **removed** from this Dataset and returned **by
        reference** (not copied) - so you can alter them and ``add()`` them back in their
        new form::

            ts = ds.pop(location="PB03B", sensor="AV319")   # taken out of ds
            ts.ts = ts.ts - 300                             # edit the live series
            ds.add(ts)                                       # put it back, changed

        Parameters:
            *predicates (Where): Predicate objects; all must match (combine with ``& | ~``).
            location (str | list, optional): The location name(s); ``~`` negates.
            variable (str | list, optional): The variable(s) being measured; ``~`` negates.
            unit (str | list, optional): Unit(s) of the measurement; ``~`` negates.
            **kwargs (str | list): Other timeseries attributes to match (e.g., sensor).

        Returns:
            Timeseries | Dataset: A single Timeseries if exactly one match is removed, a
                new Dataset of them if several match, or an empty Dataset if none match
                (in which case nothing is removed).
        """
        keep = self._matcher(predicates, location, variable, unit, kwargs)

        popped: list[T | None] = []
        remaining: list[T | None] = []
        for ts in self.timeseries:
            (popped if keep(ts) else remaining).append(ts)

        self.timeseries = remaining

        if not popped:
            return Dataset()
        if len(popped) == 1:
            return popped[0]
        return Dataset(timeseries=popped)

    def _matcher(
        self,
        predicates: tuple,
        location: str | list | None,
        variable: str | list | None,
        unit: str | list | None,
        kwargs: dict,
    ) -> Any:
        """Build the ``keep(ts)`` predicate shared by :meth:`filter` and :meth:`pop`.

        A timeseries is kept when it matches every keyword filter (``~`` negation
        included) and every positional :class:`Where` predicate. ``None`` entries never
        match.
        """
        keywords = {"location": location, "variable": variable, "unit": unit, **kwargs}
        tests = [
            Where(**{attr: value})
            for attr, value in keywords.items()
            if value is not None
        ]
        tests.extend(predicates)

        def keep(ts: T | None) -> bool:
            return ts is not None and all(test(ts) for test in tests)

        return keep

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
        facet: str = "variable",
        variable: str | list | None = None,
        ncols: int = 5,
        sharex: bool = False,
        include_outliers: bool = False,
        plot_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, list] | dict[str, tuple[Figure, list]]:
        """Plot the dataset's timeseries, in one of two layouts.

        - ``facet="variable"`` (default): one subplot per variable (pressure,
          temperature, ...), every location's series overlaid on that axis. Returns
          ``(fig, axes)`` where ``axes`` is a list (one per variable).
        - ``facet="location"``: a **separate figure per variable**, each a grid with one
          panel per location (``ncols`` wide). Every location gets a panel - left empty
          if it has no (or empty) series for that variable - and unused trailing cells are
          hidden. Multiple sensors at a location are overlaid in the same panel, and a
          legend (labelled by **sensor serial**) is shown only then; single-series panels
          get no legend. Panels are titled by location and carry no x-label (the dates are
          on the shared/rotated ticks). Returns ``{variable: (fig, axes)}``.

        Parameters:
            facet (str): ``"variable"`` or ``"location"``.
            variable (str | list, optional): restrict to these variable(s); default is
                every unique variable in the dataset.
            ncols (int): panels per row for the ``facet="location"`` grid.
            sharex (bool): for ``facet="location"``, share the x-axis across all panels so
                every row and column is aligned to the same (full) time span - the
                longest-running series sets the extent, and empty panels span it too.
            include_outliers (bool): Whether to include outliers in the plot.
            plot_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.plot().
            legend_kwargs (dict[str, Any] | None): kwargs passed to matplotlib.axes.Axes.legend().

        Returns:
            ``(fig, axes)`` for ``facet="variable"``; a ``{variable: (fig, axes)}`` dict
            for ``facet="location"``.
        """
        variables = (
            [variable]
            if isinstance(variable, str)
            else list(variable)
            if variable is not None
            else sorted({ts.variable for ts in self.timeseries if ts is not None})
        )

        if facet == "variable":
            return self._plot_by_variable(
                variables, include_outliers, plot_kwargs, legend_kwargs
            )
        if facet == "location":
            return self._plot_by_location(
                variables, ncols, sharex, include_outliers, plot_kwargs, legend_kwargs
            )

        message = f"facet must be 'variable' or 'location', got {facet!r}."
        raise ValueError(message)

    def _plot_by_variable(
        self,
        variables: list,
        include_outliers: bool,
        plot_kwargs: dict[str, Any] | None,
        legend_kwargs: dict[str, Any] | None,
    ) -> tuple[Figure, list]:
        """One subplot per variable, every location overlaid (see :meth:`plot`)."""
        fig, axs = plt.subplots(
            len(variables),
            1,
            figsize=(10, 5 * len(variables)),
            sharex=True,
            squeeze=False,
        )
        axes = list(axs.ravel())
        for ax, var in zip(axes, variables, strict=False):
            for ts in self.timeseries:
                if ts is not None and ts.variable == var and len(ts.ts) > 0:
                    ts.plot(
                        include_outliers=include_outliers,
                        ax=ax,
                        plot_kwargs=plot_kwargs,
                        legend_kwargs=legend_kwargs,
                    )
            ax.set_title(f"Timeseries for {var.capitalize()}")
            ax.set_xlabel("Time")
        fig.tight_layout()
        return fig, axes

    def _series_at(self, location: str, variable: str) -> list:
        """Non-empty timeseries at a given location and variable."""
        return [
            ts
            for ts in self.timeseries
            if ts is not None
            and ts.location == location
            and ts.variable == variable
            and len(ts.ts) > 0
        ]

    def _draw_location_panel(
        self,
        ax: Axes,
        series: list,
        include_outliers: bool,
        plot_kwargs: dict[str, Any],
        legend_kwargs: dict[str, Any],
    ) -> None:
        """Draw one location panel: overlay its series, style ticks, legend if shared."""
        for ts in series:
            ax.plot(ts.ts.index, ts.ts.to_numpy(), label=ts.sensor, **plot_kwargs)
            if include_outliers and ts.outliers is not None and len(ts.outliers) > 0:
                ax.scatter(ts.outliers.index, ts.outliers, color="red", s=5)
        ax.tick_params(labelsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        if len(series) > 1:  # only label sensors when they share a panel
            ax.legend(**legend_kwargs)

    def _plot_by_location(
        self,
        variables: list,
        ncols: int,
        sharex: bool,
        include_outliers: bool,
        plot_kwargs: dict[str, Any] | None,
        legend_kwargs: dict[str, Any] | None,
    ) -> dict[str, tuple[Figure, list]]:
        """A grid of one panel per location, a figure per variable (see :meth:`plot`)."""
        locations = self.get_locations()
        nrows = (len(locations) + ncols - 1) // ncols if locations else 1
        pkw = {"lw": 0.7, **(plot_kwargs or {})}
        lkw = {"fontsize": 7, **(legend_kwargs or {})}
        results: dict[str, tuple[Figure, list]] = {}
        for var in variables:
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(4 * ncols, 2.3 * nrows),
                squeeze=False,
                sharex=sharex,
            )
            axes = list(axs.ravel())
            for ax, loc in zip(axes, locations, strict=False):
                ax.set_title(
                    loc, fontsize=8
                )  # every location keeps a panel, even if empty
                self._draw_location_panel(
                    ax, self._series_at(loc, var), include_outliers, pkw, lkw
                )
            for ax in axes[len(locations) :]:
                ax.set_visible(False)  # hide unused trailing cells
            fig.suptitle(f"{var.capitalize()} by location", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.98))  # leave room for the suptitle
            results[var] = (fig, axes)
        return results


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

    The table is :attr:`Dataset.info` with a derived ``duration`` column appended, so
    the per-series summary has a single source.
    """

    columns: ClassVar[list[str]] = [
        "location",
        "variable",
        "sensor",
        "records",
        "start",
        "end",
        "duration",
    ]

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        table = dataset.info
        table["duration"] = table["end"] - table["start"]
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
            index: pd.DatetimeIndex | None = None
            for ts in self._dataset:
                if ts is None or ts.location != location or len(ts.ts) == 0:
                    continue
                index = ts.ts.index if index is None else index.union(ts.ts.index)
            if index is None or len(index) == 0:
                continue
            ax.broken_barh(
                _coverage_segments(index, threshold), (row - 0.4, 0.8), facecolors=color
            )

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

        # per label: key-tuple -> {sensor, records, start, end}, collapsed from the
        # dataset's `.info` table; plus the union DatetimeIndex per key, kept only for
        # the timeline plot.
        self._coverage: dict[str, dict[tuple, dict]] = {}
        self._index: dict[str, dict[tuple, pd.DatetimeIndex]] = {}
        for label, dataset in datasets.items():
            self._coverage[label] = self._summarise(dataset.info)
            index_by_key: dict[tuple, pd.DatetimeIndex] = {}
            for ts in dataset:
                if ts is None or len(ts.ts) == 0:
                    continue
                k = tuple(getattr(ts, attr) for attr in self.key)
                index_by_key[k] = (
                    ts.ts.index
                    if k not in index_by_key
                    else index_by_key[k].union(ts.ts.index)
                )
            self._index[label] = {
                k: idx.sort_values() for k, idx in index_by_key.items()
            }

        self.keys = sorted({k for cov in self._coverage.values() for k in cov})
        self.table = self._build_table()

    def _summarise(self, info: pd.DataFrame) -> dict[tuple, dict]:
        """Collapse a :attr:`Dataset.info` table into one summary row per comparison
        ``key`` (the key columns must be present in ``info``). Timeseries sharing a key
        are merged: sensors joined, records summed, span widened to the outer bounds."""
        summary: dict[tuple, dict] = {}
        for row in info.itertuples(index=False):
            k = tuple(getattr(row, attr) for attr in self.key)
            entry = summary.setdefault(
                k, {"sensors": set(), "records": 0, "start": row.start, "end": row.end}
            )
            entry["sensors"].add(row.sensor)
            entry["records"] += int(row.records)
            entry["start"] = min(entry["start"], row.start)
            entry["end"] = max(entry["end"], row.end)
        return {
            k: {
                "sensor": "+".join(sorted(s for s in v["sensors"] if s)) or None,
                "records": v["records"],
                "start": v["start"],
                "end": v["end"],
            }
            for k, v in summary.items()
        }

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
                    cov.get(k, defaults).get(metric, defaults[metric])
                    for k in self.keys
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
                index = self._index[label].get(k)
                if index is None or len(index) == 0:
                    continue
                y = row - 0.4 + j * sub_h
                ax.broken_barh(
                    _coverage_segments(index, threshold),
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
        message = (
            "Pass two or more datasets to diff(), e.g. diff({'a': ds1, 'b': ds2})."
        )
        raise TypeError(message)
    if not isinstance(datasets, dict):
        datasets = {f"ds{i}": d for i, d in enumerate(datasets)}
    return CoverageDiff(datasets, key=key)
