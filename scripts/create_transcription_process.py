import argparse
import os
import sys

from sqlalchemy import func, inspect, not_
from sqlalchemy.exc import (DataError, DBAPIError, IntegrityError,
                            StatementError)

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (PATH_LENGTH, TranscriptionInput,
                                               TranscriptionMetadata,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.grls_id import (find_subject_id,
                                              looks_like_subject_id)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.pdf_files import is_pdf_column, is_pdf_path

"""
Populates the transcription process log table with files to process and their related metadata.
"""


# The columns that hold file paths, by table.
PATH_COLUMNS = {
    TranscriptionInput.__tablename__: ['input_file'],
    TranscriptionOutput.__tablename__: ['ocr_output_file', 'list_replacement_output_file', 'pii_scrubber_output_file',
                                        'pii_scrubber_confidence_file'],
}


def _path_column_widths(session) -> dict:
    """
    Gets how many characters each file path column holds in the database.

    A database created before the columns were widened keeps its old widths, since installing the schema doesn't
    change existing columns, so this reads the live tables rather than trusting the model.

    Parameters
    ----------
    session : Session
        An open session.

    Returns
    -------
    dict
        "table.column" to width, PATH_LENGTH for a column whose width the database doesn't report.

    Raises
    ------
    RuntimeError
        If the tables don't exist, with how to create them.
    """
    inspector = inspect(session.get_bind())
    widths = {}
    for table, names in PATH_COLUMNS.items():
        if not inspector.has_table(table):
            raise RuntimeError(f"The {table} table doesn't exist. Create the tables first: "
                               f"python scripts/database_setup.py install")
        for column in inspector.get_columns(table):
            if column['name'] in names:
                widths[f"{table}.{column['name']}"] = getattr(column['type'], 'length', None) or PATH_LENGTH
    return widths


def _is_row_error(error: Exception) -> bool:
    """
    Tells whether an error raised while saving one file is about that row only, so the file can be skipped.

    Data and integrity errors (a value the column can't hold, a name that isn't valid in the column's character
    set, a duplicate) are. Errors that end the whole transaction or connection, such as a deadlock or a lost
    connection, aren't: the rows saved so far are gone with it, so the run must stop rather than carry on.

    Parameters
    ----------
    error : Exception
        The error.

    Returns
    -------
    bool
        True if only this row is affected.
    """
    if isinstance(error, (DataError, IntegrityError, UnicodeError)):
        return True
    # A value the driver couldn't encode is raised as a StatementError that isn't a database error.
    driver_error = isinstance(error, StatementError) and not isinstance(error, DBAPIError)
    return driver_error and isinstance(error.orig, UnicodeError)


def do_create_transcription_process(args: argparse.Namespace) -> None:
    """
    Creates rows for a new transcription process.

    Notes
    -----
    These input assets may then be picked up by future processes.
    Only PDFs are registered (see vmrt_tesseract_utilities.pdf_files), since they're all the OCR step reads; the
    EMR folders also hold image viewer exports, lock files and the like. Each file is saved on its own, so one that
    can't be stored (a path too long for the column, say) is reported and skipped, and the rest are still
    registered.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    stdout_logger.info(f'Creating new transcription process with strategy "{args.document_type}".')
    registered = already_registered = not_pdf = too_long = failed = 0
    root = os.path.abspath(args.path_to_describe)
    sessionmaker = get_database_session(echo=args.debug_sql)
    try:
        with sessionmaker.begin() as session:
            widths = _path_column_widths(session)
            width = widths['transcription_input.input_file']
            narrow = {column: w for column, w in widths.items() if w < PATH_LENGTH}
            if narrow:
                listed = ', '.join(f'{column} ({w})' for column, w in narrow.items())
                stdout_logger.warning(
                    f'These path columns hold fewer than {PATH_LENGTH} characters, as the database was created '
                    f'before they were widened: {listed}. Longer input paths are skipped here, and a long output '
                    f'path would fail in a later step. To widen them, start from a fresh database or see "If you '
                    f'keep a database between runs" in the README.'
                )
            registered_paths = {
                path for (path,) in (
                    session.query(TranscriptionInput.input_file)
                    .filter(TranscriptionInput.document_type == args.document_type)
                    .all()
                )
            }
            # The same rows the README's cleanup removes: not PDFs, and never transcribed.
            leftover = (
                session.query(func.count(TranscriptionInput.id.distinct()))
                .outerjoin(TranscriptionInput.assets)
                .filter(TranscriptionInput.document_type == args.document_type)
                .filter(TranscriptionOutput.id.is_(None))
                .filter(not_(is_pdf_column(TranscriptionInput.input_file)))
                .scalar()
            )
            if leftover:
                stdout_logger.warning(
                    f'{leftover} files already registered as "{args.document_type}" aren\'t PDFs, from a registration '
                    f'that took every file. OCR skips them; to remove them, see "If you keep a database between runs" '
                    f'in the README.'
                )
            for subdir, dirs, files in os.walk(root):
                dog_id = find_subject_id(subdir)
                if not dog_id:
                    # Folders with no ID at all (the top folder, for example) are skipped quietly, but one whose name
                    # looks like an ID that isn't accepted, e.g. "094-0001312", is reported so its files aren't lost.
                    if looks_like_subject_id(os.path.basename(subdir)):
                        stdout_logger.warning(
                            f'Skipping folder, its name contains a number that is not a valid subject ID: {subdir}'
                        )
                    continue
                for file in files:
                    filepath = subdir + os.sep + file
                    # Only the part below the root is checked, so a root folder named "._..." doesn't hide everything.
                    if not is_pdf_path(os.path.relpath(filepath, root).replace(os.sep, '/')):
                        not_pdf += 1
                        continue
                    if filepath in registered_paths:
                        stdout_logger.info(f"Already registered, skipping: {filepath}")
                        already_registered += 1
                        continue
                    if len(filepath) > width:
                        stdout_logger.warning(f'Not registered, the path is longer than {width} characters: {filepath}')
                        too_long += 1
                        continue
                    stdout_logger.info(f'Adding to process: {filepath}')
                    try:
                        # A savepoint per file, so one that the database rejects is rolled back on its own.
                        with session.begin_nested():
                            input = TranscriptionInput(input_file=filepath, document_type=args.document_type)
                            # This is an easy place to create the associated metadata record.
                            # However, we will only have one piece of information to add right now.
                            session.add(TranscriptionMetadata(subject_id=dog_id, transcription_input=input))
                    except Exception as e:
                        if not _is_row_error(e):
                            raise
                        error = getattr(e, 'orig', None) or e
                        stdout_logger.warning(
                            f'Not registered, the database rejected it ({type(error).__name__}: {error}): {filepath}')
                        failed += 1
                        continue
                    registered_paths.add(filepath)
                    registered += 1
    except Exception as e:
        if registered or already_registered or not_pdf or too_long or failed:
            stdout_logger.error(f'Registration failed and nothing from this run was saved. It had got as far as '
                                f'{registered} new PDFs, {already_registered} already registered, {not_pdf} files '
                                f'that aren\'t PDFs, {too_long} paths too long to store and {failed} rejected.')
        else:
            stdout_logger.error(f'Registration failed before looking at any files: {e}')
        raise
    stdout_logger.info(f'All done! Registered {registered} new PDFs. Skipped {already_registered} already '
                       f'registered, {not_pdf} files that aren\'t PDFs, {too_long} with paths too long to store, '
                       f'and {failed} the database rejected.')


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        description='Registers the PDFs in a directory of medical records for processing.')
    parser.add_argument('path_to_describe', help='The path to the EMRs we want to describe.')
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type we want to produce, document, page or block.')
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


if __name__ == '__main__':
    provided_args = parse_args()
    do_create_transcription_process(provided_args)
