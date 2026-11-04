import argparse
import os
import sys

from sqlalchemy import inspect

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (PATH_LENGTH, TranscriptionInput,
                                               TranscriptionMetadata,
                                               get_database_session)
from vmrt_tesseract_utilities.grls_id import (find_subject_id,
                                              looks_like_subject_id)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.pdf_files import is_pdf_path

"""
Populates the transcription process log table with files to process and their related metadata.
"""


def _input_file_width(session) -> int:
    """
    Gets how many characters the database's input_file column holds.

    A database created before the column was widened keeps its old width, since installing the schema doesn't
    change existing columns, so this reads the live table rather than trusting the model.

    Parameters
    ----------
    session : Session
        An open session.

    Returns
    -------
    int
        The column's width, or PATH_LENGTH if the database doesn't report one.
    """
    for column in inspect(session.get_bind()).get_columns(TranscriptionInput.__tablename__):
        if column['name'] == 'input_file':
            return getattr(column['type'], 'length', None) or PATH_LENGTH
    return PATH_LENGTH


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
    sessionmaker = get_database_session(echo=args.debug_sql)
    try:
        with sessionmaker.begin() as session:
            width = _input_file_width(session)
            if width < PATH_LENGTH:
                stdout_logger.warning(
                    f'The database\'s input_file column holds {width} characters, not {PATH_LENGTH}: it was created '
                    f'before the column was widened. Longer paths will be skipped. To store them, start from a fresh '
                    f'database or widen the columns (see "If you keep a database between runs" in the README).'
                )
            registered_paths = {
                path for (path,) in (
                    session.query(TranscriptionInput.input_file)
                    .filter(TranscriptionInput.document_type == args.document_type)
                    .all()
                )
            }
            leftover = sum(1 for path in registered_paths if not is_pdf_path(path))
            if leftover:
                stdout_logger.warning(
                    f'{leftover} files already registered as "{args.document_type}" aren\'t PDFs, from a registration '
                    f'that took every file. OCR skips them; to remove them, see "If you keep a database between runs" '
                    f'in the README.'
                )
            for subdir, dirs, files in os.walk(os.path.abspath(args.path_to_describe)):
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
                    if not is_pdf_path(filepath):
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
                        error = getattr(e, 'orig', None) or e
                        stdout_logger.warning(
                            f'Not registered, the database rejected it ({type(error).__name__}: {error}): {filepath}')
                        failed += 1
                        continue
                    registered_paths.add(filepath)
                    registered += 1
    except Exception:
        stdout_logger.error(f'Registration failed and nothing from this run was saved. It had got as far as '
                            f'{registered} new PDFs, {already_registered} already registered, {not_pdf} files '
                            f'that aren\'t PDFs, {too_long} paths too long to store and {failed} rejected.')
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
