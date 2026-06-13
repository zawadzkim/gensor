# Changelog

All notable changes to gensor are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.0]

### Added

- **`Dataset.filter` negation and predicates.** A leading `~` on any value negates it
  (`filter(location="~PB16D")` drops that location); positive and negated values may be
  mixed within an attribute and across attributes. For conditions the per-attribute
  keywords can't express, a composable `Where` predicate (`&`, `|`, `~`) can be passed
  positionally, e.g. `filter(~Where(location="PB03B", sensor="AV319"))` to drop only that
  sensor at that location.
- **`Dataset.pop(...)`** — same selection as `filter`, but removes the matched timeseries
  from the dataset and returns them by reference, for an extract → edit → `add()`
  round-trip.
- **`Dataset.info`** — a per-timeseries metadata table (location, variable, sensor,
  records, start, end). It is now the single source for the per-series summary used by
  `Dataset.coverage` (which adds a derived `duration`) and `gensor.diff` / `CoverageDiff`.
- **`Dataset.coverage`** — coverage summary that renders as a table and offers
  `.plot()` for a per-location coverage timeline (bars broken on gaps > `max_gap`).
- **`gensor.diff` / `Dataset.diff`** — coverage comparison of two or more datasets,
  aligned by key, rendering as a wide table with a `.plot()` N-way timeline.
- **`Dataset.plot(facet="location", ...)`** — a grid of one panel per location (a separate
  figure per variable), with `ncols`, `sharex` (align all panels to the full time span),
  per-panel legends only when several sensors share a panel (labelled by serial), and the
  compact grid styling. `facet="variable"` (overlaid locations) remains the default.
- **`Dataset.loc[start:end]`** — label-based selection forwarded to every timeseries,
  returning a new sliced Dataset.
- **`water_column(...)`** (and `Compensator.water_column`) — the barometric-compensation
  step exposed on its own: it returns the water column above the sensor (m) without adding
  the sensor altitude. `compensate(...)` now builds on it.
- Parser reads location/serial from labelled Diver-Office header fields; `Dataset`
  supports `(location, variable[, unit])` tuple indexing, `__contains__`, and `one()`.

### Changed

- **Compensation out-of-water filtering is now a signed cutoff and always applied.**
  The water column above a submerged sensor is physically non-negative, so negative values
  are always dropped and the near-zero out-of-water band at or below `threshold_wc` is
  removed. `threshold_wc` now defaults to `0.025` m (25 mm); set a smaller value to keep
  shallower columns, or `0` to drop only negatives. (Previously the cutoff compared the
  *absolute* value, which wrongly retained large-magnitude negatives, and was off by
  default.)

### Fixed

- `Dataset.to_sql` made robust; van Essen parser made robust.

### Removed

- The `exclude=` dict argument of `Dataset.filter` — superseded by `~` negation and
  `Where` predicates.
