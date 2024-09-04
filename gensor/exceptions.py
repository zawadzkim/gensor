class InvalidMeasurementTypeError(ValueError):
    """Raised when a timeseries of a wrong measurement type is operated upon."""

    def __init__(self, timeseries_name: str, expected_type: str = "pressure") -> None:
        self.timeseries_name = timeseries_name
        self.expected_type = expected_type
        message = f"Timeseries '{self.timeseries_name}' must be of measurement type '{self.expected_type}'."
        super().__init__(message)


class InvalidInputTypeError(TypeError):
    """Raised when an input is of the wrong type."""

    def __init__(self, input_name, expected_type, actual_type, message=None) -> None:
        self.input_name = input_name

        if isinstance(expected_type, tuple):
            self.expected_type = expected_type
            expected_types_str = " | ".join([t.__name__ for t in self.expected_type])
        else:
            self.expected_type = (expected_type,)
            expected_types_str = self.expected_type[0].__name__

        self.actual_type = actual_type

        if message is None:
            message = (
                f"Invalid type for '{self.input_name}': "
                f"Expected type '{expected_types_str}', "
                f"but got type '{self.actual_type.__name__}'."
            )
        super().__init__(message)


class InvalidInputType(TypeError):
    """Raised when a timeseries is a wrong instance type."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MissingInputError(ValueError):
    """Raised when a required input is missing."""

    def __init__(self, input_name, message=None):
        self.input_name = input_name
        if message is None:
            message = f"Missing required input: '{self.input_name}'."
        super().__init__(message)


class DatabaseNotFound(FileExistsError):
    def __init__(self, *args: object, message=None) -> None:
        message = "Database directory does not exist."
        super().__init__(message, *args)


class TimeseriesUnequal(ValueError):
    """Raised when Timeseries objects are compared and are unequal."""

    def __init__(self, *args: object, message=None) -> None:
        message = (
            "Timeseries objects must have the same location, sensor, variable, and \
        unit to be added together."
        )
        super().__init__(message, *args)


class IndexOutOfRangeError(IndexError):
    """Custom exception raised when an index is out of range in the dataset."""

    def __init__(self, index, dataset_size):
        super().__init__(
            f"Index {index} is out of range for the dataset with {dataset_size} timeseries."
        )


class TimeseriesNotFound(ValueError):
    def __init__(self, *args: object, message=None) -> None:
        message = "No matching timeseries found for the given criteria."
        super().__init__(message, *args)


class NoFilesToLoad(FileNotFoundError):
    def __init__(self, *args: object, message=None) -> None:
        message = "Directory contains no files or only contains other folders."
        super().__init__(message, *args)
