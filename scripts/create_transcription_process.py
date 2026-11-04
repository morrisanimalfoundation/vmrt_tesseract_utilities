import argparse
import os
import sys

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (PATH_LENGTH, TranscriptionInput,
                                               TranscriptionMetadata,
                                               get_database_session)
from vmrt_tesseract_utilities.grls_id import (find_subject_id,
                                              looks_like_subject_id)
from vmrt_tesseract_utilities.logging import stdout_logger

"""
Populates the transcription process log table with files to process and their related metadata.
"""


def do_create_transcription_process(args: argparse.Namespace) -> None:
    """
    Creates rows for a new transcription process.

    Notes
    -----
    These input assets may then be picked up by future processes.
    Only PDFs are registered, since they're all the OCR step reads; the EMR folders also hold image viewer exports,
    lock files and the like. A path longer than the database can store is reported and skipped, rather than
    stopping the whole registration.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    stdout_logger.info(f'Creating new transcription process with strategy "{args.document_type}".')
    assets = []
    registered = already_registered = not_pdf = 0
    too_long = []
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker.begin() as session:
        registered_paths = {
            path for (path,) in (
                session.query(TranscriptionInput.input_file)
                .filter(TranscriptionInput.document_type == args.document_type)
                .all()
            )
        }
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
                if not file.lower().endswith('.pdf'):
                    not_pdf += 1
                    continue
                filepath = subdir + os.sep + file
                if filepath in registered_paths:
                    stdout_logger.info(f"Already registered, skipping: {filepath}")
                    already_registered += 1
                    continue
                if len(filepath) > PATH_LENGTH:
                    too_long.append(filepath)
                    continue
                stdout_logger.info(f'Adding to process: {filepath}')
                input = TranscriptionInput(input_file=filepath, document_type=args.document_type)
                # This is an easy place to create the associated metadata record.
                # However, we will only have one piece of information to add right now.
                metadata = TranscriptionMetadata(subject_id=dog_id, transcription_input=input)
                assets.append(input)
                assets.append(metadata)
                registered_paths.add(filepath)
                registered += 1
        session.add_all(assets)
    for filepath in too_long:
        stdout_logger.warning(f'Not registered, the path is longer than {PATH_LENGTH} characters: {filepath}')
    stdout_logger.info(f'All done! Registered {registered} new PDFs. Skipped {already_registered} already '
                       f'registered, {not_pdf} files that aren\'t PDFs, and {len(too_long)} with paths too long '
                       f'to store.')


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
