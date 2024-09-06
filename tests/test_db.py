from sqlalchemy import text

from gensor import read_from_sql


def test_db_creation_default_location(db):
    """Test that the database is created in the default location."""

    with db as con:
        query = text("SELECT name FROM sqlite_master WHERE type='table';")
        result = con.execute(query).fetchall()
        assert result == [], "No tables should exist in the newly created database."

    assert (
        db.db_directory / db.db_name
    ).exists(), "Database file should be created in the default location."


def test_db_create_table(db, timeseries):
    ts = timeseries[0]

    schema_name = f"{ts.location}_{ts.sensor}_{ts.variable}_{ts.unit}"

    def check_number_of_tables():
        query = text("SELECT name FROM sqlite_master WHERE type='table';")
        return con.execute(query).fetchall()

    with db as con:
        db.create_table(schema_name, ts.variable)
        assert (
            len(check_number_of_tables()) == 1
        ), "There should be 1 table in the database."

    with db as con:
        db.create_table(schema_name, ts.variable)
        assert (
            len(check_number_of_tables()) == 1
        ), "There should still be 1 table in the database. Creation of the second one should be skipped"


def test_save_and_load_timeseries(db, timeseries):
    """Test saving and loading Timeseries from an in-memory database."""
    ts = timeseries[0]

    message = ts.to_sql(db)
    assert "table updated" in message

    loaded_ts = read_from_sql(
        db=db,
        load_all=False,
        location=ts.location,
        sensor=ts.sensor,
        variable=ts.variable,
        unit=ts.unit,
    )

    assert ts == loaded_ts, "Loaded timeseries should match the saved timeseries"


def test_save_and_load_dataset(db, timeseries):
    """Test saving and loading Dataset from the database."""

    timeseries.to_sql(db)

    read_from_sql(
        db=db,
        load_all=True,
    )
