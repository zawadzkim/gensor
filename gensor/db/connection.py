"""Module for database connection."""

from pathlib import Path

import pydantic as pyd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..exceptions import DatabaseNotFound


class DatabaseConnection(pyd.BaseModel):
    """Class for handling the database connection.
    If no database exists at the specified path, it will be created.
    If no database is specified, an in-memory database will be used.

    The user should specify the database directory and name separately. If directory is not specified,
    current directory and a default name are used. ."""

    model_config = pyd.ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    in_memory: bool = False
    db_directory: Path = Path.cwd()
    db_name: str = "gensor.db"
    engine: Engine | None = None
    session: Session | None = None

    def __post_init__(self):
        self.connect()

    def _verify_path(self) -> str:
        if self.in_memory:
            return "sqlite:///:memory:"
        else:
            if not self.db_directory.exists():
                raise DatabaseNotFound()
            else:
                return f"sqlite:///{self.db_directory}/{self.db_name}"

    def connect(self):
        sqlite_path = self._verify_path()

        self.engine = create_engine(sqlite_path)
        session = sessionmaker(bind=self.engine)
        self.session = session()

        return session()
