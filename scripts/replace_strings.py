import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

from sqlalchemy import and_

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

"""
Replaces strings in scrubbed text files from a CSV or TSV.
"""


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
    """
    input_file = None
    try:
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

        replacer = StringReplacer(strings_to_replace, parsed_args.replacement_string)
        scrubbed_text = replacer.replace(orig_text)

        with open(output_file, "w") as outfile:
            outfile.write(scrubbed_text)
        stdout_logger.info(f"Scrubbed file written to {output_file}")

        output_log.list_replacement_output_file = output_file
        return output_log
    except Exception as e:
        stdout_logger.error(f"An error occurred while processing {input_file}: {e}\n{traceback.format_exc()}")
        return None


def scrub_and_write_files(process_filepath_data: List[object], strings_to_replace: List[str], parsed_args: argparse.Namespace, use_multiprocessing: bool = True) -> None:
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
    """
    session_maker = get_database_session(echo=parsed_args.debug_sql)
    batch_size = parsed_args.chunk_size
    success = True
    for i in range(0, len(process_filepath_data), batch_size):
        batch = process_filepath_data[i:i + batch_size]
        with session_maker.begin() as session:
            if use_multiprocessing:
                with ProcessPoolExecutor(max_workers=parsed_args.max_workers) as executor:
                    results = executor.map(process_file_wrapper, batch, [strings_to_replace] * len(batch), [parsed_args] * len(batch))
                    for output_log in results:
                        if output_log:
                            session.add(output_log)
                            session.add(StringReplacementCompletion(output_id=output_log.id, pass_id=parsed_args.replacement_pass_id))
                        else:
                            success = False
            else:
                for output_log in batch:
                    result = process_file_wrapper(output_log, strings_to_replace, parsed_args)
                    if result:
                        session.add(result)
                        session.add(StringReplacementCompletion(output_id=result.id, pass_id=parsed_args.replacement_pass_id))
                    else:
                        success = False
    if not success:
        # If any files failed to process, exit with an error code.
        sys.exit(1)


def process_file_wrapper(output_log: object, strings_to_replace: List[str], parsed_args: argparse.Namespace) -> Optional[object]:
    """
    Processes one transcription output.

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
    """
    # session_maker = get_database_session(echo=parsed_args.debug_sql)
    # with session_maker.begin() as session:
    #     session.add(output_log)
    #     return process_file(output_log, strings_to_replace, parsed_args)
    return process_file(output_log, strings_to_replace, parsed_args)


def get_files_to_process(args: argparse.Namespace) -> list:
    """
    Gets a list of input files to process.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    results: list
      The list of input files.
    """
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker() as session:
        query = (session.query(TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 .outerjoin(StringReplacementCompletion,
                            and_(StringReplacementCompletion.output_id == TranscriptionOutput.id,
                                 StringReplacementCompletion.pass_id == args.replacement_pass_id))
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file != None)  # noqa: E711
                 .where(StringReplacementCompletion.id.is_(None))
                 .order_by(TranscriptionOutput.id)
                 .limit(args.chunk_size)
        )
        results = query.all()
    stdout_logger.info(f"Found {len(results)} files for this replacement pass.")
    
    return results


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
    effective_targets = StringReplacer(
        target_strings, replacement_string
    ).target_strings

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
    # parser.add_argument("--offset", type=int, default=0, help="The number of records to skip before beginning processing.")
    parser.add_argument("--debug-sql", action="store_true", help="Enable SQL debugging")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing for debugging")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes for multiprocessing")
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


if __name__ == "__main__":
    args = parse_args()
    try:
        target_strings = read_target_strings(args.data_file, args.key_column)
        pass_hash, effective_targets = make_pass_hash(target_strings, args.replacement_string)
        args.replacement_pass_id = get_or_create_pass(args, pass_hash, len(effective_targets))
        results = get_files_to_process(args)
        scrub_and_write_files(results, target_strings, args, not args.no_multiprocessing)
    except Exception as e:
        stdout_logger.error(f"Error in main execution: {e}\n{traceback.format_exc()}")
        # If an error occurs, exit with an error code.
        sys.exit(1)
