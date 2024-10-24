import logging

from .analysis.outliers import OutlierDetection
from .core.dataset import Dataset
from .core.timeseries import Timeseries
from .io.read import read_from_csv, read_from_sql
from .processing.compensation import Compensator, compensate
from .processing.transform import Transformation

__all__ = [
    # basic data types
    "Dataset",
    "Timeseries",
    # data transformation
    "OutlierDetection",
    "Transformation",
    "Compensator",
    "compensate",
    # getters
    "read_from_csv",
    "read_from_sql",
]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
