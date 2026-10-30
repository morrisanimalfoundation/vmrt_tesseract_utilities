import pytest

from vmrt_tesseract_utilities.database import get_database_session

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
