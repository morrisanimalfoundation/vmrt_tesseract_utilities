"""
Utility functions for scrubbing actions.
"""
import argparse

from vmrt_tesseract_utilities.database import get_database_session, TranscriptionOutput, TranscriptionInput
from vmrt_tesseract_utilities.logging import stdout_logger


def write_scrubbed_txt(output_filename: str, anonymized_text: str) -> None:
    """
    Writes the anonymized text to an output file.

    Parameters
    ----------
    output_filename : str
        The path to the file.
    anonymized_text : str
        The anonymized text.
    """
    try:
        if anonymized_text:
            with open(output_filename, 'w') as f:
                f.write(anonymized_text)
    except Exception as e:
        stdout_logger.error(f'Error writing scrubbed output: {e}')
        raise


def get_files_to_process(args: argparse.Namespace) -> list:
    """
    Retrieves a list of input files to process from the database.

    This function queries the database to fetch a batch of transcription
    output records that meet specific criteria:

    - Belong to the document type specified in the arguments.
    - Have an OCR output file.
    - Have not yet been processed by the PII scrubber.

    The query result is limited to a chunk size and offset, allowing for
    batch processing of large datasets.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments containing the following attributes:

        - debug_sql : bool
            Flag to enable or disable SQL debugging output.
        - document_type : str
            The type of document to filter by.
        - chunk_size : int
            The maximum number of records to fetch.
        - offset : int
            The offset from which to start fetching records.

    Returns
    -------
    list
        A list of `TranscriptionOutput` objects representing the files
        to be processed.
    """
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker.begin() as session:
        query = (
            session.query(TranscriptionOutput)
            .outerjoin(TranscriptionInput.assets)
            .where(TranscriptionInput.document_type == args.document_type)
            .where(TranscriptionOutput.ocr_output_file != None)
            .where(TranscriptionOutput.pii_scrubber_output_file == None)
            .limit(args.chunk_size)
            .offset(args.offset)
        )
        return query.all()
