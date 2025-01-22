import argparse

from vmrt_tesseract_utilities import database
from vmrt_tesseract_utilities.logging import stdout_logger

"""
Database utilities.
"""


def perform_database_action(args: argparse.Namespace) -> None:
    """
    Performs a database action based on the provided arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    if args.operation not in ('install', 'drop'):
        raise ValueError("'operation' must be either 'install' or 'drop'")
    if args.operation == 'install':
        stdout_logger.info('Installing database tables.')
        database_action_install(args)
    if args.operation == 'drop':
        stdout_logger.info('Dropping all database tables.')
        database_action_drop(args)
    stdout_logger.info('All done!')


def database_action_install(args: argparse.Namespace) -> None:
    """
    Installs the database by creating all required tables.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    session = database.get_database_session(echo=args.debug_sql)
    database.Base.metadata.create_all(session.kw.get('bind'))


def database_action_drop(args: argparse.Namespace) -> None:
    """
    Drops all tables from the database.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    session = database.get_database_session(echo=args.debug_sql)
    database.Base.metadata.drop_all(bind=session.kw.get('bind'))


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Performs database utility related functions.',)
    parser.add_argument('operation', help='The database operation to perform, install or drop.', choices=['install', 'drop'])
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    perform_database_action(provided_args)
