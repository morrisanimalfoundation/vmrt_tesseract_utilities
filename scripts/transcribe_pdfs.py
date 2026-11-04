import argparse
import os
import sys
from typing import LiteralString

from sqlalchemy import func

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.tesseract_operations import (
    OcrSetupError, TesseractOperationBlock, TesseractOperationDoc,
    TesseractOperationPage, check_ocr_setup)

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


def _output_path(base_path: str, db_input: TranscriptionInput, ocr_result: dict) -> str:
    """
    Builds the path of the text file for one OCR result.

    The name starts with the input's id, so PDFs with the same file name in different folders (such as two
    dogs' "Medical Records.pdf") don't overwrite each other's text.

    Parameters
    ----------
    base_path: str
      The output directory.
    db_input: TranscriptionInput
      The input the result came from.
    ocr_result: dict
      The OCR result, with its page and block numbers.

    Returns
    -------
    output_path: str
      The path to write the result to.
    """
    name = os.path.splitext(os.path.basename(db_input.input_file))[0]
    return f'{base_path}/{db_input.id}-{name}-{ocr_result["page"]}-{ocr_result["block"]}.txt'


def parse_shard(value: str) -> tuple[int, int]:
    """
    Parses a `--shard` value: "i/N" means this run takes the inputs whose ID leaves remainder i when divided by N.

    Several runs started with the same N and every i from 0 to N - 1 split the inputs between them. Unlike
    `--offset`, the split doesn't move while other runs finish files, so no file is picked twice or skipped.

    Parameters
    ----------
    value: str
        The value, e.g. "0/4".

    Returns
    -------
    tuple[int, int]
        (i, N).

    Raises
    ------
    argparse.ArgumentTypeError
        If the value isn't "i/N" with 0 <= i < N.

    Examples
    --------
    >>> parse_shard('2/4')
    (2, 4)
    >>> parse_shard('4/4')
    Traceback (most recent call last):
        ...
    argparse.ArgumentTypeError: '4/4': the shard number must be from 0 to 3.
    """
    try:
        index, count = (int(part) for part in value.split('/'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}': expected i/N, e.g. 0/4.")
    if count < 1:
        raise argparse.ArgumentTypeError(f"'{value}': the number of shards must be at least 1.")
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"'{value}': the shard number must be from 0 to {count - 1}.")
    return index, count


def run_tesseract(args: argparse.Namespace) -> None:
    """
    Runs a Tesseract strategy over the pages of a PDF.

    Notes
    -----
    The process is documented in the `process_output` table.
    Each input's `ocr_status` is set to 'success' or 'failed', with the error saved in `ocr_error`.
    Failed inputs are skipped on later runs unless `--retry-failed` is passed.
    With `--shard i/N`, only inputs whose ID leaves remainder i when divided by N are processed.
    Setup problems that would fail every file stop the run before any file is touched.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Raises
    ------
    OcrSetupError
        If poppler or Tesseract isn't usable.
    """
    base_path = _get_and_create_output_directory(args)
    if args.document_type not in ('document', 'page', 'block',):
        raise ValueError("'--document-type' must be either 'document', 'page' or 'block'")
    check_ocr_setup()
    stdout_logger.info(f'Here we go with document type: {args.document_type}')
    if args.document_type == 'document':
        op = TesseractOperationDoc()
    if args.document_type == 'page':
        op = TesseractOperationPage()
    if args.document_type == 'block':
        op = TesseractOperationBlock()
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker() as session:
        query = (session.query(TranscriptionInput, TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 # input_file has a case-sensitive collation, and file extensions can be in any case.
                 .where(func.lower(TranscriptionInput.input_file).like('%.pdf'))
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file == None))  # noqa: E711
        if not args.retry_failed:
            query = query.where(TranscriptionInput.ocr_status.is_distinct_from('failed'))
        if args.shard:
            index, count = args.shard
            query = query.where(TranscriptionInput.id % count == index)
        # Without an order, which rows --offset skips isn't defined and can change between runs.
        query = (query.order_by(TranscriptionInput.id, TranscriptionOutput.id)
                 .limit(args.chunk_size)
                 .offset(args.offset))
        count = query.count()
        stdout_logger.info(f'Found {count} files to transcribe.')
        failures = []
        for db_input, db_output in query.all():
            db_output_logs = []
            try:
                output = op.process_row(db_input.input_file)
                # Write every text file before touching the database, so a write error (disk full, permissions)
                # is recorded as a failure like any other, rather than stopping the run.
                output_paths = []
                for ocr_result in output:
                    output_path = _output_path(base_path, db_input, ocr_result)
                    _write_file(output_path, ocr_result['content'])
                    output_paths.append(output_path)
            except Exception as e:
                error = f'{type(e).__name__}: {e}'
                stdout_logger.error(f'Failed to transcribe {db_input.input_file}: {error}')
                failures.append((db_input.input_file, error))
                db_input.ocr_status = 'failed'
                db_input.ocr_error = error
                # Commit the failure too, so it's recorded even if a later file crashes the run.
                session.commit()
                continue
            db_input.ocr_status = 'success'
            db_input.ocr_error = None
            for ocr_result, output_path in zip(output, output_paths):
                # Each page or block gets its own row; an existing empty row is filled by the first one.
                if db_output is None:
                    db_output = TranscriptionOutput()
                db_output.ocr_output_file = output_path
                db_output.ocr_confidence = ocr_result['confidence']
                db_output.transcription_input = db_input
                db_output_logs.append(db_output)
                db_output = None
            session.add_all(db_output_logs)
            # Commit per file, so a crash part way through a chunk only loses the file in progress.
            session.commit()
    _log_run_summary(count, failures)


def _log_run_summary(count: int, failures: list) -> None:
    """
    Logs how many files were transcribed and lists each failure.

    Parameters
    ----------
    count: int
      The number of files the run attempted.
    failures: list
      (input_file, error) tuples for the files that failed.
    """
    stdout_logger.info(f'All done! Transcribed {count - len(failures)} of {count} files.')
    if failures:
        stdout_logger.warning(f'{len(failures)} file(s) failed:')
        for input_file, error in failures:
            stdout_logger.warning(f'  {input_file}: {error}')


def parse_args() -> argparse.Namespace:
    """
    Calculates Tesseract confidence scores for a list of file describing JSON objects, outputting modified JSON.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(description='Run Tesseract OCR over the PDFs queued in the database.')
    parser.add_argument('output_to', help='Path to the output directory.')
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type we want to produce, document, page or block.')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='The number of records to process.')
    parser.add_argument('--offset', type=int, default=0,
                        help='The number of records to skip before beginning processing.')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Also process files that failed on an earlier run.')
    parser.add_argument('--shard', type=parse_shard, metavar='i/N',
                        help='Only process inputs whose ID leaves remainder i when divided by N, so N runs started '
                             'with 0/N to N-1/N can transcribe in parallel without picking the same file.')
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


if __name__ == '__main__':
    provided_args = parse_args()
    try:
        run_tesseract(provided_args)
    except OcrSetupError as e:
        stdout_logger.error(f'Stopping before transcribing anything: {e}')
        sys.exit(1)
