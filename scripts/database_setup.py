import argparse

from vmrt_tesseract_utilities import database

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
        database_action_install(args)
    if args.operation == 'drop':
        database_action_drop(args)


def database_action_install(args: argparse.Namespace) -> None:
    """
    Installs the database by creating all required tables.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    engine = database.get_engine(echo=args.debug_sql)
    database.Base.metadata.create_all(engine)


def database_action_drop(args: argparse.Namespace) -> None:
    """
    Drops all tables from the database.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    engine = database.get_engine(echo=args.debug_sql)
    database.Base.metadata.drop_all(bind=engine)


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
    parser.add_argument('operation')
    parser.add_argument('--debug-sql', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    perform_database_action(provided_args)
