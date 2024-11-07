from sqlalchemy import text

from gensor import read_from_sql


def test_db_creation_default_location(db):
    """Test that the database is created in the default location."""

    with db as con:
        query = text("SELECT name FROM sqlite_master WHERE type='table';")
        result = con.execute(query).fetchall()
        assert len(result) == 1, (
            "There should only be 1 table in the database (metadata)."
        )

    assert (db.db_directory / db.db_name).exists(), (
        "Database file should be created in the default location."
    )


def test_db_create_table(db, baro_timeseries):
    ts = baro_timeseries[0]
    timestamp_start_fmt = ts.start.strftime("%Y%m%d%H%M%S")
    schema_name = f"{ts.location}_{ts.sensor}_{ts.variable}_{ts.unit}_{timestamp_start_fmt}".lower()

    def check_number_of_tables():
        query = text("SELECT name FROM sqlite_master WHERE type='table';")
        return con.execute(query).fetchall()

    with db as con:
        db.create_table(schema_name, ts.variable)
        assert len(check_number_of_tables()) == 2, (
            "There should be 2 tables in the database (metadata and newly created one)."
        )

    with db as con:
        db.create_table(schema_name, ts.variable)
        assert len(check_number_of_tables()) == 2, (
            "There should still be 2 tables in the database. Creation of the second one should be skipped"
        )


def test_save_and_load_timeseries(db, baro_timeseries):
    """Test saving and loading Timeseries from an in-memory database."""
    ts = baro_timeseries[0]

    message = ts.to_sql(db)
    assert "table and metadata updated" in message

    loaded_ts = read_from_sql(
        db=db, load_all=False, location=ts.location, variable=ts.variable, unit=ts.unit
    )

    assert ts == loaded_ts[0], "Loaded timeseries should match the saved timeseries."


def test_save_and_load_dataset(db, baro_timeseries):
    """Test saving and loading Dataset from the database."""

    baro_timeseries.to_sql(db)

    loaded_ds = read_from_sql(
        db=db,
        load_all=True,
    )

    assert len(loaded_ds.timeseries) == len(baro_timeseries), (
        f"The loaded dataset should match the saved dataset. {len(loaded_ds.timeseries)} vs {len(baro_timeseries)}"
    )


def test_locate_table_by_attributes(db, baro_timeseries):
    """Test locating a table by basic attributes and extra keys."""

    ts = baro_timeseries[0]
    extra_info = {"sensor": ts.sensor}

    ts.to_sql(db)

    located_table = db.get_timeseries_metadata(
        location=ts.location, variable=ts.variable, unit=ts.unit, **extra_info
    )

    assert located_table["location"].iloc[0] == ts.location, (
        f"Expected to sensor name '{ts.location}', but got '{located_table[0][2]}'"
    )
    assert located_table["extra"].iloc[0]["sensor"] == ts.sensor, (
        f"Expected to sensor name '{ts.sensor}', but got '{located_table[0][7]}'"
    )
