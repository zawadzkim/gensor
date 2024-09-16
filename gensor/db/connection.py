"""Module defining database connection object.

Classes:
    DatabaseConnection: Database connection object
"""

from pathlib import Path
from typing import Any

import pydantic as pyd
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)

from ..exceptions import DatabaseNotFound


class DatabaseConnection(pyd.BaseModel):
    """Database connection object.
    If no database exists at the specified path, it will be created.
    If no database is specified, an in-memory database will be used."""

    model_config = pyd.ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    metadata: MetaData = MetaData()
    db_directory: Path = Path.cwd()
    db_name: str = "gensor.db"
    engine: Engine | None = None

    def _verify_path(self) -> str:
        """Verify database path."""

        if not self.db_directory.exists():
            raise DatabaseNotFound()
        return f"sqlite:///{self.db_directory}/{self.db_name}"

    def connect(self) -> Connection:
        """Connect to the database and initialize the engine.
        If engine is None > create it with verified path > reflect.
        After connecting, ensure the timeseries_metadata table is present.
        """
        if self.engine is None:
            sqlite_path = self._verify_path()
            self.engine = create_engine(sqlite_path)

        connection = self.engine.connect()

        self.create_metadata()

        return connection

    def dispose(self) -> None:
        """Dispose of the engine, closing all connections."""
        if self.metadata:
            self.metadata.clear()
        if self.engine:
            self.engine.dispose()

    def __enter__(self) -> Connection:
        """Enable usage in a `with` block by returning the engine."""
        con = self.connect()
        if self.engine:
            self.metadata.reflect(bind=self.engine)
        return con

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Dispose of the engine when exiting the `with` block."""
        self.dispose()

    def get_tables(self) -> list | None:
        """Return the list of tables, excluding the 'timeseries_metadata' table."""
        with self:
            tables = self.metadata.tables

            if not tables:
                print("This database has no tables.")
                return None
            else:
                filtered_tables = [
                    table for table in tables if table != "__timeseries_metadata__"
                ]
                return filtered_tables

    def create_metadata(self) -> str | Table:
        """Create a metadata table if it doesn't exist yet and store ts metadata."""

        metadata_table = Table(
            "__timeseries_metadata__",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("table_name", String, unique=True),
            Column("location", String),
            Column("sensor", String),
            Column("variable", String),
            Column("unit", String),
            Column("logger_alt", Float, nullable=True),
            Column("location_alt", Float, nullable=True),
            Column("timestamp_start", String, nullable=True),
            Column("timestamp_end", String, nullable=True),
        )

        if self.engine:
            metadata_table.create(self.engine, checkfirst=True)
            self.metadata.reflect(bind=self.engine)
            return metadata_table
        else:
            return "Engine does not exist."

    def create_table(self, schema_name: str, column_name: str) -> Table | str:
        """Create a table in the database.

        Schema name is a string representing the location, sensor, variable measured and
        unit of measurement. This is a way of preserving the metadata of the Timeseries.
        The index is always `timestamp` and the column name is dynamicly create from
        the measured variable.
        """

        if schema_name in self.metadata.tables:
            return self.metadata.tables[schema_name]

        ts_table = Table(
            schema_name,
            self.metadata,
            Column("timestamp", String, primary_key=True),
            Column(column_name, Float),
            info={},
        )

        if self.engine:
            ts_table.create(self.engine, checkfirst=True)
            self.metadata.reflect(bind=self.engine)
            return ts_table
        else:
            return "Engine does not exist."
