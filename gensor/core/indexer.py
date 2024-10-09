from __future__ import annotations


import pandas as pd


from typing import Any


class TimeseriesIndexer:
    """A wrapper for the Pandas indexers (e.g., loc, iloc) to return Timeseries objects."""

    # marked indexer as Any to silence mypy. BaseIndexer is normally not indexable:

    def __init__(self, parent: "Timeseries", indexer: Any):
        self.parent = parent
        self.indexer = indexer

    def __getitem__(self, key: str) -> "Timeseries":
        """Allows using the indexer (e.g., loc) and wraps the result in a Timeseries."""

        result = self.indexer[key]

        if isinstance(result, pd.Series):
            return self.parent.model_copy(update={"ts": result}, deep=True)
        message = f"Expected pd.Series, but got {type(result)} instead."
        raise TypeError(message)