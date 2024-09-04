import os

import pytest

from gensor.db import DatabaseConnection


@pytest.fixture
def in_memory_db():
    db_connection = DatabaseConnection(in_memory=True)
    yield db_connection
    db_connection.session.close()


@pytest.fixture
def file_based_db_default_location():
    db_connection = DatabaseConnection(in_memory=False)
    yield db_connection
    db_connection.session.close()

    os.remove(db_connection.db_directory / db_connection.db_name)


@pytest.fixture
def file_based_db_specified_location(tmp_path):
    db_directory = tmp_path
    db_connection = DatabaseConnection(in_memory=False, db_directory=db_directory)
    yield db_connection
    db_connection.session.close()
