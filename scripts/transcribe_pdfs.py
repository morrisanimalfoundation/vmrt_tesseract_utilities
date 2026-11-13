import argparse
import os
import re
import sys
from typing import LiteralString

from vmrt_tesseract_utilities.cli import (add_max_failure_rate_argument,
                                          normalize_flag_spelling,
                                          run_exit_code)
from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import log_run_summary, stdout_logger
from vmrt_tesseract_utilities.pdf_files import is_pdf_column
from vmrt_tesseract_utilities.tesseract_operations import (
    OcrSetupError, TesseractOperationBlock, TesseractOperationDoc,
    TesseractOperationPage, check_ocr_setup)

"""
Runs Tesseract on items stored in the input database.
"""

# The values of TranscriptionInput.ocr_status. A PDF is STARTED while it's being read, RETRYING while it's being
# read again after a run was cut short on it, and SUCCESS or FAILED once it's done. NULL means it hasn't been tried.
STATUS_STARTED = 'started'
STATUS_RETRYING = 'retrying'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'

# The error saved for a PDF whose OCR was cut short twice, so a PDF that always gets its run killed (usually for
# running out of memory) is skipped like any other failure instead of stopping every later run.
INTERRUPTED_TWICE_ERROR = ('OcrInterrupted: OCR of this PDF was cut short twice before it could finish: its process '
                           'was killed, most likely for running out of memory, or the run was stopped. Try it again '
                           'with --retry-failed.')


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
    Nothing claims a file in the database, so this only holds while every run at the same time uses the same N:
    0/2 and 0/3 both take ID 6. The split is even when IDs go up by 1, as MySQL's do by default.

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
        If the value isn't "i/N", two whole numbers with nothing else, with 0 <= i < N.

    Examples
    --------
    >>> parse_shard('2/4')
    (2, 4)
    >>> parse_shard('4/4')
    Traceback (most recent call last):
        ...
    argparse.ArgumentTypeError: '4/4': the shard number must be from 0 to 3.
    """
    # Not int() on each part: it also takes "+1", " 1 " and "1_0".
    match = re.fullmatch(r'(\d+)/(\d+)', value)
    if not match:
        raise argparse.ArgumentTypeError(f"'{value}': expected i/N, two whole numbers such as 0/4.")
    shard_index, shard_count = int(match.group(1)), int(match.group(2))
    if shard_count < 1:
        raise argparse.ArgumentTypeError(f"'{value}': the number of shards must be at least 1.")
    if shard_index >= shard_count:
        raise argparse.ArgumentTypeError(f"'{value}': the shard number must be from 0 to {shard_count - 1}.")
    return shard_index, shard_count


def run_tesseract(args: argparse.Namespace) -> tuple[int, list[tuple[str, str]]]:
    """
    Runs a Tesseract strategy over the pages of a PDF.

    Notes
    -----
    The process is documented in the `process_output` table.
    Each input's `ocr_status` ends as 'success' or 'failed', with the error saved in `ocr_error`.
    Failed inputs are skipped on later runs unless `--retry-failed` is passed.
    While OCR runs, the input is saved as 'started', so a run that's killed part way through a PDF (by running out
    of memory, or a stopped build) leaves a record of it. A later run tries a 'started' PDF once more, as
    'retrying', and if that attempt is cut short too, marks it failed instead of trying again (unless
    `--retry-failed` is passed, which reads it). Nothing marks which run is reading a PDF, so runs that overlap
    (shards with different N) can take each other's PDF for one that was cut short.
    With `--shard i/N`, only inputs whose ID leaves remainder i when divided by N are processed.
    Setup problems that would fail every file stop the run before any file is touched.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    tuple[int, list[tuple[str, str]]]
        The number of files tried, and (input file, error) for each file that failed.

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
                 # The same rule registration uses, so the two agree on which files are PDFs.
                 .where(is_pdf_column(TranscriptionInput.input_file))
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file == None))  # noqa: E711
        if not args.retry_failed:
            query = query.where(TranscriptionInput.ocr_status.is_distinct_from(STATUS_FAILED))
        if args.shard:
            shard_index, shard_count = args.shard
            query = query.where(TranscriptionInput.id % shard_count == shard_index)
        # Without an order, which rows --offset skips isn't defined and can change between runs.
        query = (query.order_by(TranscriptionInput.id, TranscriptionOutput.id)
                 .limit(args.chunk_size)
                 .offset(args.offset))
        count = query.count()
        stdout_logger.info(f'Found {count} files to transcribe.')
        failures = []
        for db_input, db_output in query.all():
            db_output_logs = []
            if db_input.ocr_status == STATUS_RETRYING and not args.retry_failed:
                _record_failure(session, db_input, INTERRUPTED_TWICE_ERROR, failures)
                continue
            if db_input.ocr_status in (STATUS_STARTED, STATUS_RETRYING):
                stdout_logger.warning(f'Item {db_input.input_file}: an earlier run was cut short while reading it, '
                                      f'so it is tried once more.')
                db_input.ocr_status = STATUS_RETRYING
            else:
                db_input.ocr_status = STATUS_STARTED
            # An error from an earlier attempt doesn't describe this one.
            db_input.ocr_error = None
            # Saved before OCR starts, so it's still there if this run is killed part way through the PDF.
            session.commit()
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
                _record_failure(session, db_input, f'{type(e).__name__}: {e}', failures)
                continue
            db_input.ocr_status = STATUS_SUCCESS
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
    log_run_summary('Transcribed', count, failures)
    return count, failures


def _record_failure(session, db_input: TranscriptionInput, error: str, failures: list) -> None:
    """
    Logs that a PDF failed, adds it to the run's failures, and saves it as failed with its error.

    Committed straight away, so the failure is recorded even if a later file crashes the run.

    Parameters
    ----------
    session : Session
        The open database session.
    db_input : TranscriptionInput
        The PDF that failed.
    error : str
        The error, as saved in `ocr_error`.
    failures : list
        The run's (input_file, error) failures, added to.
    """
    stdout_logger.error(f'Failed to transcribe {db_input.input_file}: {error}')
    failures.append((db_input.input_file, error))
    db_input.ocr_status = STATUS_FAILED
    db_input.ocr_error = error
    session.commit()


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
    add_max_failure_rate_argument(parser)
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


def main() -> int:
    """
    Transcribes the queued PDFs, and works out the run's exit code.

    A PDF that fails doesn't stop the run, so the exit code is how anything checking it (Jenkins) finds out that
    every PDF, or more than --max-failure-rate of them, failed.

    Returns
    -------
    int
        0 if the run succeeded, 1 if the OCR tools aren't usable, or EXIT_TOO_MANY_FAILURES if too many PDFs
        failed.
    """
    provided_args = parse_args()
    try:
        count, failures = run_tesseract(provided_args)
    except OcrSetupError as e:
        stdout_logger.error(f'Stopping before transcribing anything: {e}')
        return 1
    return run_exit_code(count, len(failures), provided_args.max_failure_rate)


if __name__ == '__main__':
    sys.exit(main())
