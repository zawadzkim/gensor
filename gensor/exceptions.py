class InvalidMeasurementTypeError(ValueError):
    """Raised when a timeseries of a wrong measurement type is operated upon."""

    def __init__(self, expected_type: str = "pressure") -> None:
        self.expected_type = expected_type
        message = f"Timeseries must be of measurement type '{self.expected_type}'."
        super().__init__(message)


class MissingInputError(ValueError):
    """Raised when a required input is missing."""

    def __init__(self, input_name: str, message: str | None = None) -> None:
        self.input_name = input_name
        if message is None:
            message = f"Missing required input: '{self.input_name}'."
        super().__init__(message)


class DatabaseNotFound(FileExistsError):
    def __init__(self, *args: object, message: str | None = None) -> None:
        message = "Database directory does not exist."
        super().__init__(message, *args)


class TimeseriesUnequal(ValueError):
    """Raised when Timeseries objects are compared and are unequal."""

    def __init__(self, *args: object, message: str | None = None) -> None:
        message = (
            "Timeseries objects must have the same location, sensor, variable, and \
        unit to be added together."
        )
        super().__init__(message, *args)


class IndexOutOfRangeError(IndexError):
    """Custom exception raised when an index is out of range in the dataset."""

    def __init__(self, index: int, dataset_size: int) -> None:
        super().__init__(
            f"Index {index} is out of range for the dataset with {dataset_size} timeseries."
        )


class TimeseriesNotFound(ValueError):
    def __init__(self, *args: object, message: str | None = None) -> None:
        message = "No matching timeseries found for the given criteria."
        super().__init__(message, *args)


class NoFilesToLoad(FileNotFoundError):
    def __init__(self, *args: object, message: str | None = None) -> None:
        message = "Directory contains no files or only contains other folders."
        super().__init__(message, *args)
