from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TimeseriesIndexer:
    """A wrapper for the Pandas indexers (e.g., loc, iloc) to return Timeseries objects."""

    # marked indexer as Any to silence mypy. BaseIndexer is normally not indexable:
    # the same for the `parent`. It should by always type Timeseries, but I don't want
    # to deal with circular imports just for type hints for the devs...

    def __init__(self, parent: Any, indexer: Any):
        self.parent = parent
        self.indexer = indexer

    def __getitem__(self, key: str) -> Any:
        """Allows using the indexer (e.g., loc) and wraps the result in a Timeseries."""

        result = self.indexer[key]

        if isinstance(result, pd.Series):
            return self.parent.model_copy(update={"ts": result}, deep=True)

        if isinstance(result, (int | float | str | pd.Timestamp | np.float64)):
            return result

        message = f"Expected pd.Series, but got {type(result)} instead."
        raise TypeError(message)
