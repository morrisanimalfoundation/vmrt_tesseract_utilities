"""
Replaces strings in scrubbed text files from a CSV or TSV.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import stat
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional

from sqlalchemy import and_, or_
from sqlalchemy.pool import NullPool

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (StringReplacementCompletion,
                                               StringReplacementPass,
                                               TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.string_replacer import StringReplacer

# Increment this when the replacement matching behavior changes
REPLACEMENT_ALGORITHM_VERSION = "1"

# Exit code when the run finished but some files failed or had missing output, so CI (e.g. Jenkins) can tell it
# apart from the script crashing (exit code 1).
EXIT_INCOMPLETE = 3

# How many failed or missing files to list in the log; the rest are counted.
MAX_LISTED = 20

# The StringReplacer built in this process, with the inputs it was built from. Building one compiles a pattern for
# every target, so get_replacer builds it once and reuses it instead of rebuilding it for every file.
_replacer_cache: Optional[tuple] = None


def read_target_strings(data_file: str, key_column: str) -> List[str]:
    """
    Reads target strings from a CSV or TSV file.

    Parameters
    ----------
    data_file : str
        The path to the CSV or TSV file.
    key_column : str
        The name of the column containing the target strings.

    Returns
    -------
    list
        A list of strings extracted from the specified column.

    Raises
    ------
    ValueError
        If the provided file is not a CSV or TSV, or if the specified key column is not found.
    FileNotFoundError
        If the specified data file does not exist.
    """
    strings = []
    # Detect file type based on extension
    if data_file.endswith(".csv"):
        delimiter = ","
    elif data_file.endswith(".tsv"):
        delimiter = "\t"
    else:
        raise ValueError("Invalid data file format. The data_file must be a CSV or TSV.")
    try:
        # Read the target strings from the specified column.
        with open(data_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            if key_column not in reader.fieldnames:
                raise ValueError(f"Key column '{key_column}' not found in data file.")
            strings.extend([row[key_column] for row in reader])
        return strings
    except FileNotFoundError as e:
        stdout_logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        stdout_logger.error(f"An error occurred in read_target_strings: {e}\n{traceback.format_exc()}")
        raise


def get_replacer(strings_to_replace: Iterable[str], replacement_string: str) -> StringReplacer:
    """
    Gets the StringReplacer for these targets and replacement, building it only the first time in this process.

    Building a replacer compiles a pattern for every target, which can take longer than replacing the targets in a
    small file, so it's reused for every file after the first instead of being rebuilt for each one. Each worker
    process builds its own, unless the parent built it first and the workers inherited it when they were forked.

    The inputs are compared by value, so a call with different targets or a different replacement gets a new
    replacer.

    Parameters
    ----------
    strings_to_replace : iterable of str
        The strings to search for.
    replacement_string : str
        The string to replace them with.

    Returns
    -------
    StringReplacer
        The replacer for these inputs.
    """
    global _replacer_cache
    # Read once into a tuple, so the key and the replacer agree even if the targets are a one-shot iterable.
    targets = tuple(strings_to_replace)
    key = (targets, replacement_string)
    if _replacer_cache is None or _replacer_cache[0] != key:
        _replacer_cache = (key, StringReplacer(list(targets), replacement_string))
    return _replacer_cache[1]


def process_file(output_log, strings_to_replace: List[str], parsed_args: argparse.Namespace) -> Optional[object]:
    """
    Processes a single file, replacing text and writing outputs.

    Parameters
    ----------
    output_log : TranscriptionOutput
        An object containing file data.
    strings_to_replace : list
        The list of strings to search for.
    parsed_args : argparse.Namespace
        The parsed args.

    Returns
    -------
    TranscriptionOutput
        The updated file data object.

    Raises
    ------
    Exception
        Any error reading or writing the file, for the caller to record.
    """
    if hasattr(output_log, 'list_replacement_output_file') and output_log.list_replacement_output_file:
        input_file = output_log.list_replacement_output_file
        output_file = input_file
    else:
        input_file = output_log.ocr_output_file
        input_filename = os.path.basename(str(input_file))
        filename_without_extension = os.path.splitext(input_filename)[0]
        output_dir = f"{parsed_args.output_dir}/list_replacement_output_file/{parsed_args.document_type}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{filename_without_extension}.txt"

    with open(str(input_file), "r") as f:
        orig_text = f.read()

    replacer = get_replacer(strings_to_replace, parsed_args.replacement_string)
    scrubbed_text = replacer.replace(orig_text)

    _write_atomically(output_file, scrubbed_text)
    stdout_logger.info(f"Scrubbed file written to {output_file}")

    output_log.list_replacement_output_file = output_file
    return output_log


def _write_atomically(output_file: str, text: str) -> None:
    """
    Writes text to a file so the file always holds either its old contents or the new ones, never part of either.

    Later passes write over the only copy of the text with every earlier pass applied, so a crash or a full disk
    part way through a plain write would lose it.

    Parameters
    ----------
    output_file : str
        The file to write.
    text : str
        The text to write.
    """
    directory = os.path.dirname(os.path.abspath(output_file))
    # The temporary file is in the same directory, so os.replace swaps it in with a single rename.
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(text)
            # On disk before the rename, so a power cut can't leave the renamed file empty.
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_path, _new_file_mode(output_file))
        os.replace(tmp_path, output_file)
    except BaseException:
        # Also covers a full disk, which often only shows up when the buffered text is flushed.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _new_file_mode(output_file: str) -> int:
    """
    Gets the permissions to give a rewritten file: its current ones if it exists, otherwise the ones a plain
    open() would give it. mkstemp creates temporary files readable only by their owner.

    Parameters
    ----------
    output_file : str
        The file about to be written.

    Returns
    -------
    int
        The permission bits.
    """
    if os.path.exists(output_file):
        return stat.S_IMODE(os.stat(output_file).st_mode)
    # os.umask can only be read by setting it, so set it back straight away.
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def scrub_and_write_files(process_filepath_data: List[object], strings_to_replace: List[str], parsed_args: argparse.Namespace, use_multiprocessing: bool = True) -> List[tuple]:
    """
    Processes a list of files, replacing text and writing outputs in parallel.

    Parameters
    ----------
    process_filepath_data : list
        A list of file data objects.
    strings_to_replace : list
        The list of strings to search for.
    parsed_args : argparse.Namespace
        The parsed args.
    use_multiprocessing : bool
        Whether to use multiprocessing or not.

    Returns
    -------
    list of tuple
        (output_id, input_file, error) for each file that failed.
    """
    batch_size = parsed_args.chunk_size
    failures = []
    for i in range(0, len(process_filepath_data), batch_size):
        batch = process_filepath_data[i:i + batch_size]
        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=parsed_args.max_workers) as executor:
                results = list(executor.map(process_file_wrapper, batch, [strings_to_replace] * len(batch), [parsed_args] * len(batch)))
        else:
            results = [process_file_wrapper(output_log, strings_to_replace, parsed_args) for output_log in batch]
        failures.extend(result for result in results if result is not None)
    return failures


def process_file_wrapper(output_log: object, strings_to_replace: List[str], parsed_args: argparse.Namespace) -> Optional[tuple]:
    """
    Processes one transcription output and records the result, success or failure, in its own transaction.

    Parameters
    ----------
    output_log : TranscriptionOutput
        An object containing file data.
    strings_to_replace : list
        The list of strings to search for.
    parsed_args : argparse.Namespace
        The parsed args.

    Returns
    -------
    tuple or None
        (output_id, input_file, error) if the file failed, otherwise None.
    """
    # This runs once per file, so don't keep a connection open for each one.
    session_maker = get_database_session(echo=parsed_args.debug_sql, poolclass=NullPool)
    with session_maker.begin() as session:
        session.add(output_log)
        # Read before the commit, which expires the object's attributes.
        output_id = output_log.id
        input_file = output_log.list_replacement_output_file or output_log.ocr_output_file
        try:
            process_file(output_log, strings_to_replace, parsed_args)
            error = None
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
            stdout_logger.error(f"An error occurred while processing {input_file}: {e}\n{traceback.format_exc()}")
        _record_result(session, output_id, parsed_args.replacement_pass_id, error)
    return (output_id, str(input_file), error) if error else None


def _record_result(session, output_id: int, pass_id: int, error: Optional[str]) -> None:
    """
    Saves the result of an output's attempt at a pass, replacing the result of any earlier attempt.

    Parameters
    ----------
    session : Session
        The open database session.
    output_id : int
        The transcription output's id.
    pass_id : int
        The replacement pass's id.
    error : str or None
        The error if the attempt failed, otherwise None.
    """
    completion = (session.query(StringReplacementCompletion)
                  .filter_by(output_id=output_id, pass_id=pass_id)
                  .one_or_none())
    if completion is None:
        completion = StringReplacementCompletion(output_id=output_id, pass_id=pass_id)
        session.add(completion)
    completion.status = 'failed' if error else 'success'
    completion.error = error
    completion.completed_at = datetime.datetime.now()


def get_files_to_process(args: argparse.Namespace, failed_this_run: frozenset = frozenset()) -> list:
    """
    Gets a list of input files to process.

    Outputs that haven't been attempted for this pass are selected. Outputs that failed it are only selected with
    --retry-failed, and then not the ones that already failed in this run, so a file that keeps failing can't be
    picked again and again.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    failed_this_run: frozenset
        The ids of the outputs that failed earlier in this run.

    Returns
    -------
    results: list
      The list of input files.
    """
    # With --until-done this runs once per chunk, so don't leave a pool of connections open each time.
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    not_attempted = StringReplacementCompletion.id.is_(None)
    if args.retry_failed:
        selected = or_(not_attempted, StringReplacementCompletion.status == 'failed')
        if failed_this_run:
            selected = and_(selected, TranscriptionOutput.id.notin_(failed_this_run))
    else:
        selected = not_attempted
    with sessionmaker() as session:
        query = (session.query(TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 .outerjoin(StringReplacementCompletion,
                            and_(StringReplacementCompletion.output_id == TranscriptionOutput.id,
                                 StringReplacementCompletion.pass_id == args.replacement_pass_id))
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file != None)  # noqa: E711
                 .where(selected)
                 .order_by(TranscriptionOutput.id)
                 .limit(args.chunk_size))
        results = query.all()
    stdout_logger.info(f"Found {len(results)} files for this replacement pass.")
    return results


def find_missing_outputs(args: argparse.Namespace) -> List[tuple]:
    """
    Finds outputs that completed this pass but whose replacement file no longer exists.

    The run reports them and ends with EXIT_INCOMPLETE, rather than finding nothing to do and looking successful
    while later stages point at missing files.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    list of tuple
        (output_id, path) for each missing file. The path is None if the output has none recorded.
    """
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    with sessionmaker() as session:
        rows = list(
            session.query(TranscriptionOutput.id, TranscriptionOutput.list_replacement_output_file)
            .join(TranscriptionOutput.transcription_input)
            .join(StringReplacementCompletion, StringReplacementCompletion.output_id == TranscriptionOutput.id)
            .where(TranscriptionInput.document_type == args.document_type)
            .where(StringReplacementCompletion.pass_id == args.replacement_pass_id)
            .where(StringReplacementCompletion.status == 'success')
            .order_by(TranscriptionOutput.id)
        )
    return [(output_id, path) for output_id, path in rows if not path or not os.path.exists(path)]


def count_failed(args: argparse.Namespace) -> int:
    """
    Counts the outputs whose last attempt at this pass failed, in this run or an earlier one.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    int
        The number of failed outputs.
    """
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    with sessionmaker() as session:
        return (session.query(StringReplacementCompletion)
                .join(TranscriptionOutput, StringReplacementCompletion.output_id == TranscriptionOutput.id)
                .join(TranscriptionOutput.transcription_input)
                .where(TranscriptionInput.document_type == args.document_type)
                .where(StringReplacementCompletion.pass_id == args.replacement_pass_id)
                .where(StringReplacementCompletion.status == 'failed')
                .count())


def make_pass_hash(
    target_strings: List[str],
    replacement_string: str,
) -> tuple[str, List[str]]:
    """
    Creates a hash for the effective string-replacement configuration.

    Parameters
    ----------
    target_strings : list of str
        The replacement targets read from the CSV or TSV.
    replacement_string : str
        The string to replace each target with.

    Returns
    ----------
    tuple
        A tuple containing the SHA-256 pass hash and the effective ordered targets.
    """
    # Built through get_replacer so the files can reuse it: forked workers inherit it, and a run without
    # multiprocessing uses it as it is. The list is a copy, so changing the returned one can't change that replacer.
    effective_targets = list(get_replacer(target_strings, replacement_string).target_strings)

    payload = json.dumps(
        {
            "version": REPLACEMENT_ALGORITHM_VERSION,
            "targets": effective_targets,
            "replacement_string": replacement_string,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest(), effective_targets


def get_or_create_pass(args, pass_hash: str, target_count: int) -> int:
    """
    Gets the database record for a replacement pass or creates it if needed.

    Parameters:
        args : argparse.Namespace
            The parsed command-line arguments.
        pass_hash : str
            The hash identifying the effective replacement configuration.
        target_count : int
            The number of effective target strings in the pass.

    Returns:
        int
            The database ID of the replacement pass.
    """
    sessionmaker = get_database_session(echo=args.debug_sql)

    with sessionmaker.begin() as session:
        replacement_pass = (
            session.query(StringReplacementPass)
            .filter_by(pass_hash=pass_hash)
            .one_or_none()
        )

        if replacement_pass is None:
            replacement_pass = StringReplacementPass(
                pass_hash=pass_hash,
                replacement_string=args.replacement_string,
                source_file=args.data_file,
                key_column=args.key_column,
                target_count=target_count,
            )
            session.add(replacement_pass)
            session.flush()

        return replacement_pass.id


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args : argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(description="Replace strings in scrubbed text files from a CSV or TSV.")
    parser.add_argument("data_file", help="Path to the CSV or TSV file containing the strings.")
    parser.add_argument("key_column", help="Name of the column in the CSV/TSV containing the keys.")
    parser.add_argument("replacement_string", help="The string to replace the keys with.")
    parser.add_argument("output_dir", help="Path to the output directory.")
    parser.add_argument("--document-type", type=str, default="document", help="The document type we want to produce, document, page or block.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="The number of records to process.")
    parser.add_argument("--debug-sql", action="store_true", help="Enable SQL debugging")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing for debugging")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes for multiprocessing")
    parser.add_argument("--until-done", action="store_true", help="Keep processing chunks until no files are left, instead of stopping after one.")
    parser.add_argument("--retry-failed", action="store_true", help="Also process files that failed this pass on an earlier run.")
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


def run_pass(args: argparse.Namespace, target_strings: List[str]) -> int:
    """
    Runs the replacement pass over one chunk of files, or every chunk with --until-done, and reports the result.

    Files that fail don't stop the run: they're listed at the end and the run returns EXIT_INCOMPLETE, so an
    unattended job keeps making progress and still shows that the pass isn't finished. Once no files are left to
    process, the run also checks that every completed output still has its file, and reports any that don't.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args, including the pass id.
    target_strings : list of str
        The strings to replace.

    Returns
    -------
    int
        The exit code: 0 if everything succeeded, EXIT_INCOMPLETE if some files failed or are missing.
    """
    processed = 0
    failures = []
    while True:
        files = get_files_to_process(args, frozenset(output_id for output_id, _, _ in failures))
        if not files:
            break
        processed += len(files)
        failures.extend(scrub_and_write_files(files, target_strings, args, not args.no_multiprocessing))
        if not args.until_done:
            break
    # Checking every completed output is only worth it when nothing is left to do, which is when a missing file
    # would otherwise make the pass look finished. A full chunk means more files may be waiting.
    nothing_left = not files or len(files) < args.chunk_size
    missing = find_missing_outputs(args) if nothing_left else []
    # Includes files that failed on earlier runs and were skipped, so the pass isn't reported as done until
    # they're fixed or retried.
    failed_total = count_failed(args)
    _log_summary(args, processed, failures, failed_total, missing, nothing_left)
    return EXIT_INCOMPLETE if failed_total or missing else 0


def _log_summary(args: argparse.Namespace, processed: int, failures: List[tuple], failed_total: int,
                 missing: List[tuple], checked_missing: bool) -> None:
    """
    Logs how many files were processed, and lists the failed and missing files with what to do about them.

    The full list of missing files is also written to a report in the output directory, since it can be long.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    processed : int
        The number of files this run attempted.
    failures : list of tuple
        (output_id, input_file, error) for each file that failed in this run.
    failed_total : int
        The number of files currently marked as failed for this pass, including from earlier runs.
    missing : list of tuple
        (output_id, path) for each completed output whose file is missing.
    checked_missing : bool
        Whether this run checked for missing files. If it did and found none, an old report is removed.
    """
    stdout_logger.info(f"All done! Replaced strings in {processed - len(failures)} of {processed} files.")
    if failures:
        stdout_logger.warning(f"{len(failures)} file(s) failed in this run:")
        for _, input_file, error in failures[:MAX_LISTED]:
            stdout_logger.warning(f"  {input_file}: {error}")
        if len(failures) > MAX_LISTED:
            stdout_logger.warning(f"  ... and {len(failures) - MAX_LISTED} more.")
    if failed_total:
        stdout_logger.warning(f"{failed_total} file(s) in total have failed this pass. They're skipped on later "
                              f"runs; fix the cause and run again with --retry-failed.")
    if failed_total or missing:
        stdout_logger.warning("Don't start step 4 (PII scrubbing) until this pass finishes with exit code 0: these "
                              "files still contain the strings this pass should have replaced.")
    report = os.path.join(args.output_dir, f"missing_replacement_outputs_pass_{args.replacement_pass_id}.txt")
    if checked_missing and not missing and os.path.exists(report):
        os.remove(report)
    if missing:
        lines = [f"output {output_id}: {path or '(no path recorded)'}" for output_id, path in missing]
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report, "w") as f:
            f.write("\n".join(lines) + "\n")
        stdout_logger.warning(f"{len(missing)} file(s) completed this pass, but their output file is missing "
                              f"(full list in {report}):")
        for line in lines[:MAX_LISTED]:
            stdout_logger.warning(f"  {line}")
        if len(missing) > MAX_LISTED:
            stdout_logger.warning(f"  ... and {len(missing) - MAX_LISTED} more.")
        stdout_logger.warning("A file's text is the result of every pass run on it, so it can only be rebuilt from "
                              "the OCR output: see the README's step 3 for how to reset these files and rerun the "
                              "passes.")


if __name__ == "__main__":
    args = parse_args()
    try:
        target_strings = read_target_strings(args.data_file, args.key_column)
        pass_hash, effective_targets = make_pass_hash(target_strings, args.replacement_string)
        args.replacement_pass_id = get_or_create_pass(args, pass_hash, len(effective_targets))
        exit_code = run_pass(args, target_strings)
    except Exception as e:
        stdout_logger.error(f"Error in main execution: {e}\n{traceback.format_exc()}")
        # If an error occurs, exit with an error code.
        sys.exit(1)
    sys.exit(exit_code)
