"""Module defining database connection object.

Classes:
    DatabaseConnection: Database connection object
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pydantic as pyd
from sqlalchemy import (
    JSON,
    Column,
    Connection,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    func,
)

from ..exceptions import DatabaseNotFound

logger = logging.getLogger(__name__)


class DatabaseConnection(pyd.BaseModel):
    """Database connection object.
    If no database exists at the specified path, it will be created.
    If no database is specified, an in-memory database will be used.

    Attributes
        metadata (MetaData): SQLAlchemy metadata object.
        db_directory (Path): Path to the database to connect to.
        db_name (str): Name for the database to connect to.
        engine (Engine | None): SQLAlchemy Engine instance.
    """

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

    def get_timeseries_metadata(
        self,
        location: str | None = None,
        variable: str | None = None,
        unit: str | None = None,
        **kwargs: dict,
    ) -> pd.DataFrame:
        """
        List timeseries available in the database.

        Parameters:
            location (str): Location attribute to match.
            variable (str): Variable attribute to match.
            unit (str): Unit attribute to match.
            **kwargs: Additional filters. Must match the attributes of the
                Timeseries instance user is trying to retrieve.

        Returns:
            pd.DataFrame: The name of the matching table or None if no table is found.
        """
        with self as con:
            if "__timeseries_metadata__" not in self.metadata.tables:
                logger.info("The metadata table does not exist in this database.")
                return pd.DataFrame()

            metadata_table = self.metadata.tables["__timeseries_metadata__"]

            base_filters = []

            if location is not None:
                base_filters.append(metadata_table.c.location.ilike(location))
            if variable is not None:
                base_filters.append(metadata_table.c.variable.ilike(variable))
            if unit is not None:
                base_filters.append(metadata_table.c.unit.ilike(unit))

            extra_filters = [
                func.json_extract(metadata_table.c.extra, f"$.{k}").ilike(v)
                for k, v in kwargs.items()
                if v is not None
            ]

            # True in and_(True, *arg) fixis FutureWarning of dissallowing empty
            # filters in the future.
            query = metadata_table.select().where(
                and_(True, *base_filters, *extra_filters)
            )

            result = con.execute(query).fetchall()

            return pd.DataFrame(result).set_index("id") if result else pd.DataFrame()

    def create_metadata(self) -> Table | None:
        """Create a metadata table if it doesn't exist yet and store ts metadata."""

        metadata_table = Table(
            "__timeseries_metadata__",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("table_name", String, unique=True),
            Column("location", String),
            Column("variable", String),
            Column("unit", String),
            Column("start", String, nullable=True),
            Column("end", String, nullable=True),
            Column("extra", JSON, nullable=True),
            Column("cls", String, nullable=False),
        )

        if self.engine:
            metadata_table.create(self.engine, checkfirst=True)
            self.metadata.reflect(bind=self.engine)
            return metadata_table
        else:
            logger.info("Engine does not exist.")
            return None

    def create_table(self, schema_name: str, column_name: str) -> Table | None:
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
            logger.info("Engine does not exist.")
            return None
