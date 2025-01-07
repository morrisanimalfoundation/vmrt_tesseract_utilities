import argparse
import os
import re

from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionMetadata,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger

"""
Populates the transcription process log table with files to process and their related metadata.
"""

# Pattern to match and extract the dog id.
DOG_ID_RE = r"(094-[0-9]{6})"


def do_create_transcription_process(args: argparse.Namespace) -> None:
    """
    Creates rows for a new transcription process.

    Notes
    -----
    These input assets may then be picked up by future processes.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    """
    stdout_logger.info(f'Creating new transcription process with strategy "{args.document_type}".')
    assets = []
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker.begin() as session:
        for subdir, dirs, files in os.walk(args.path_to_describe):
            dog_match = re.search(DOG_ID_RE, subdir)
            if dog_match:
                dog_id = dog_match.group(0)
            else:
                continue
            for file in files:
                filepath = subdir + os.sep + file
                stdout_logger.info(f'Adding to process: {filepath}')
                input = TranscriptionInput(input_file=filepath, document_type=args.document_type)
                # This is an easy place to create the associated metadata record.
                # However, we will only have one piece of information to add right now.
                metadata = TranscriptionMetadata(subject_id=dog_id, transcription_input=input)
                assets.append(input)
                assets.append(metadata)
        session.add_all(assets)
    stdout_logger.info('All done!')


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Creates a new transcription process by scanning a directory of medical records.',)
    parser.add_argument('path_to_describe', help='The path to the EMRs we want to describe.')
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type we want to produce, document, page or block.')
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    do_create_transcription_process(provided_args)
