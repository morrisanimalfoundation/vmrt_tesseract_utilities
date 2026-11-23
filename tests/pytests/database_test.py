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
