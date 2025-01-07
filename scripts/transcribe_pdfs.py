import argparse
import os
from typing import LiteralString

from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.tesseract_operations import (
    TesseractOperationBlock, TesseractOperationDoc, TesseractOperationPage)

"""
Runs Tesseract on items stored in the input database.
"""


def _get_and_create_output_directory(args: argparse.Namespace) -> LiteralString | str | bytes:
    """
    Gets the output directory and attempts to create it if it doesn't exist.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    base_path: str
      The path to the output directory.
    """
    base_path = os.path.join(args.output_to, 'unstructured_text', args.document_type)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    return base_path


def _write_file(output_path: str, data: str) -> None:
    """
    Writes the ocr data to an output file.

    Parameters
    ----------
    output_path: str
      The path to the output file.
    data: str
      The data to be written.
    """
    with open(output_path, 'w') as f:
        f.write(data)


def run_tesseract(args: argparse.Namespace) -> None:
    """
    Runs a Tesseract strategy over the pages of a PDF.

    Notes
    -----
    The process is documented in the `process_output` table.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    """
    base_path = _get_and_create_output_directory(args)
    if args.document_type not in ('document', 'page', 'block',):
        raise ValueError("'--document_type' must be either 'document', 'page' or 'block'")
    stdout_logger.info(f'Here we go with document type: {args.document_type}')
    if args.document_type == 'document':
        op = TesseractOperationDoc()
    if args.document_type == 'page':
        op = TesseractOperationPage()
    if args.document_type == 'block':
        op = TesseractOperationBlock()
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker.begin() as session:
        query = (session.query(TranscriptionInput, TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 .where(TranscriptionInput.input_file.like('%.pdf'))
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file == None)  # noqa: E711
                 .limit(args.chunk_size)
                 .offset(args.offset))
        count = query.count()
        stdout_logger.info(f'Found {count} files to transcribe.')
        for db_input, db_output in query.all():
            db_output_logs = []
            output = op.process_row(db_input.input_file)
            for ocr_result in output:
                bits = os.path.splitext(os.path.basename(db_input.input_file))
                output_path = f'{base_path}/{bits[0]}-{ocr_result["page"]}-{ocr_result["block"]}.txt'
                _write_file(output_path, ocr_result['content'])
                if db_output is None:
                    db_output = TranscriptionOutput()
                db_output.ocr_output_file = output_path
                db_output.ocr_confidence = ocr_result['confidence']
                db_output.transcription_input = db_input
                db_output_logs.append(db_output)
                session.add_all(db_output_logs)
    stdout_logger.info('All done!')


def parse_args() -> argparse.Namespace:
    """
    Calculates Tesseract confidence scores for a list of file describing JSON objects, outputting modified JSON.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Outputs the change in two Fisher inventory files.')
    parser.add_argument('output_to', help='Path to the output directory.')
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type we want to produce, document, page or block.')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='The number of records to process.')
    parser.add_argument('--offset', type=int, default=0,
                        help='The number of records to skip before beginning processing.')
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    run_tesseract(provided_args)
