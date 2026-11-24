import pytest
from sqlalchemy import create_engine, text

from vmrt_tesseract_utilities.database import (ADD_TEXT_SOURCE_SQL, Base,
                                               MissingColumnError,
                                               check_text_source_column,
                                               get_database_session)

"""
Tests for the database utilities.
"""


def test_get_database_session_missing_connection_string(monkeypatch):
    """
    Test that `get_database_session` fails clearly when `SQL_CONNECTION_STRING` is not set.

    Asserts:
        A RuntimeError naming the missing variable is raised.
    """
    monkeypatch.delenv('SQL_CONNECTION_STRING', raising=False)
    with pytest.raises(RuntimeError, match='SQL_CONNECTION_STRING'):
        get_database_session()


def test_get_database_session_with_connection_string(monkeypatch):
    """
    Test that `get_database_session` returns a working sessionmaker when `SQL_CONNECTION_STRING` is set.

    Asserts:
        The returned sessionmaker is bound to the configured database.
    """
    monkeypatch.setenv('SQL_CONNECTION_STRING', 'sqlite:///:memory:')
    sessionmaker = get_database_session()
    assert str(sessionmaker.kw['bind'].url) == 'sqlite:///:memory:'


def test_text_source_column_check(tmp_path):
    """
    Test that a database created before issue #82 is caught, with the SQL that fixes it, and one with the column or
    with no tables yet passes.

    Asserts:
        No error for new tables or no tables; MissingColumnError with ADD_TEXT_SOURCE_SQL for the old table, and none
        once that SQL has been run.
    """
    engine = create_engine(f'sqlite:///{tmp_path}/new.db')
    check_text_source_column(engine)
    Base.metadata.create_all(engine)
    check_text_source_column(engine)

    old = create_engine(f'sqlite:///{tmp_path}/old.db')
    with old.begin() as connection:
        connection.execute(text('CREATE TABLE transcription_output (id INTEGER PRIMARY KEY, ocr_output_file TEXT)'))
    with pytest.raises(MissingColumnError, match='ALTER TABLE transcription_output ADD COLUMN text_source'):
        check_text_source_column(old)
    with old.begin() as connection:
        connection.execute(text(ADD_TEXT_SOURCE_SQL))
    check_text_source_column(old)


def test_text_source_column_check_takes_a_sessionmaker(monkeypatch, tmp_path):
    """
    Test that the check also takes a step's sessionmaker, as the steps pass it.

    Asserts:
        MissingColumnError for a database without the column; none with it.
    """
    old = create_engine(f'sqlite:///{tmp_path}/old.db')
    with old.begin() as connection:
        connection.execute(text('CREATE TABLE transcription_output (id INTEGER PRIMARY KEY, ocr_output_file TEXT)'))
    monkeypatch.setenv('SQL_CONNECTION_STRING', f'sqlite:///{tmp_path}/old.db')
    with pytest.raises(MissingColumnError):
        check_text_source_column(get_database_session())
    monkeypatch.setenv('SQL_CONNECTION_STRING', f'sqlite:///{tmp_path}/new.db')
    Base.metadata.create_all(create_engine(f'sqlite:///{tmp_path}/new.db'))
    check_text_source_column(get_database_session())


@pytest.mark.parametrize('script', ['scripts.replace_strings', 'scripts.metadata_miners.visit_date_miner'])
def test_steps_stop_on_a_database_without_text_source(monkeypatch, tmp_path, caplog, script):
    """
    Test that steps 3 and 5 stop before doing anything on a database created before issue #82, and go ahead on one
    with the column.

    Asserts:
        check_database is False, with the SQL that fixes it logged, for the old database, and True for a new one.
    """
    import argparse
    import importlib
    import logging
    module = importlib.import_module(script)
    args = argparse.Namespace(debug_sql=False)
    old = create_engine(f'sqlite:///{tmp_path}/old.db')
    with old.begin() as connection:
        connection.execute(text('CREATE TABLE transcription_output (id INTEGER PRIMARY KEY, ocr_output_file TEXT)'))
    monkeypatch.setenv('SQL_CONNECTION_STRING', f'sqlite:///{tmp_path}/old.db')
    with caplog.at_level(logging.ERROR, logger='import_utility'):
        assert module.check_database(args) is False
    assert 'ADD COLUMN text_source' in caplog.text
    Base.metadata.create_all(create_engine(f'sqlite:///{tmp_path}/new.db'))
    monkeypatch.setenv('SQL_CONNECTION_STRING', f'sqlite:///{tmp_path}/new.db')
    assert module.check_database(args) is True
