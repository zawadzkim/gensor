# Changelog

All notable changes to gensor are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0]

### Added

- `Dataset.filter` value negation: a leading `~` drops matches (e.g. `filter(location="~PB16D")`); positive and negated values may be mixed within and across attributes.
- `Where` predicate (`&`, `|`, `~`) for combined/cross-attribute conditions, e.g. `filter(~Where(location="PB03B", sensor="AV319"))` drops only that sensor at that location.
- `Dataset.pop(...)`: same selection as `filter`, but removes the matches and returns them by reference for an extract → edit → `add()` round-trip.
- `Dataset.info`: per-timeseries metadata table (location, variable, sensor, records, start, end); now the single source for `Dataset.coverage` (which adds a derived `duration`) and `gensor.diff` / `CoverageDiff`.
- `Dataset.coverage`: coverage summary that renders as a table and offers `.plot()` for a per-location coverage timeline.
- `gensor.diff` / `Dataset.diff`: coverage comparison of two or more datasets, with a wide table and an N-way timeline `.plot()`.
- `Dataset.plot(facet="location", ncols=, sharex=)`: a grid of one panel per location (a separate figure per variable); empty locations keep an empty panel; per-panel legend (by sensor serial) only when several sensors share a panel. `facet="variable"` remains the default.
- `Dataset.loc[start:end]`: label slicing forwarded to every timeseries, returning a new sliced Dataset.
- `water_column(...)` and `Compensator.water_column`: the barometric-compensation step on its own (water column above the sensor, in metres), without adding the sensor altitude; `compensate(...)` now builds on it.
- Parser reads location/serial from labelled Diver-Office header fields; `Dataset` supports `(location, variable[, unit])` tuple indexing, `__contains__`, and `one()`.

### Changed

- Compensation out-of-water filtering is now a signed cutoff and always applied. Negative water columns are always dropped (physically impossible for a submerged sensor) and the near-zero band at or below `threshold_wc` is removed. `threshold_wc` now defaults to `0.025` m (25 mm); set a smaller value to keep shallower columns, or `0` to drop only negatives. Previously the cutoff compared the absolute value, wrongly retaining large-magnitude negatives, and was off by default.

### Fixed

- `Dataset.to_sql` made robust; van Essen parser made robust.

### Removed

- The `exclude=` dict argument of `Dataset.filter`, superseded by `~` negation and `Where` predicates.
