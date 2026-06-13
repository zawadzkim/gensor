import logging

from .core.dataset import Dataset, Where, diff
from .core.timeseries import Timeseries
from .io.read import read_from_csv, read_from_sql
from .log import set_log_level
from .processing.compensation import compensate, water_column

__all__ = [
    # basic data types
    "Dataset",
    "Timeseries",
    "Where",
    "compensate",
    "water_column",
    # comparison
    "diff",
    # getters
    "read_from_csv",
    "read_from_sql",
    "set_log_level",
]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
