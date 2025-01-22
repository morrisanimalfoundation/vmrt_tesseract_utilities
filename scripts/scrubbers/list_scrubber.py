import argparse
import csv
import os

from vmrt_tesseract_utilities.database import (
    get_database_session,
)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.scrubbing_utils import get_files_to_process
from vmrt_tesseract_utilities.string_replacer import StringReplacer


def read_target_strings(data_file: str, key_column: str) -> list:
    """
    Reads target strings from a CSV or TSV file.

    This function reads a CSV or TSV file and extracts the values from a
    specified column. These values are typically used as target strings
    for replacement or other text processing operations.

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
        If the provided file is not a CSV or TSV, or if the specified
        key column is not found.
    FileNotFoundError
        If the specified data file does not exist.
    """
    try:
        strings = []
        # Detect file type based on extension
        if data_file.endswith(".csv"):
            delimiter = ","
        elif data_file.endswith(".tsv"):
            delimiter = "\t"
        else:
            raise ValueError("Invalid data file format. Must be CSV or TSV.")

        with open(data_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            # Check if the key column exists
            if key_column not in reader.fieldnames:
                raise ValueError(f"Key column '{key_column}' not found in "
                                 f"data file.")
            strings.extend([row[key_column] for row in reader])
        return strings
    except FileNotFoundError as e:
        stdout_logger.error(f"Data file not found: {e}")
        raise  # Re-raise the exception to be handled at a higher level


def scrub_and_write_files( process_filepath_data: list, strings_to_replace: list, parsed_args: argparse.Namespace) -> None:
    """
    Processes a list of files, replacing text and writing outputs.

    Parameters
    ----------
    process_filepath_data : list
        A list of file data dictionaries.
    strings_to_replace : list
        The list of strings to search for.
    parsed_args : argparse.Namespace
        The parsed args.
    """
    sessionmaker = get_database_session(echo=parsed_args.debug_sql)
    with sessionmaker.begin() as session:
        for output_log in process_filepath_data:
            try:
                # Use the current replacement file if it already exists.
                if output_log.list_replacement_output_file:
                    input_file = output_log.list_replacement_output_file
                    output_file = input_file
                else:
                    input_file = output_log.ocr_output_file
                    input_filename = os.path.basename(str(output_log.ocr_output_file))
                    filename_without_extension = os.path.splitext(input_filename)[0]
                    output_dir = (
                        f"{parsed_args.output_dir}/list_replacement_output_file/{parsed_args.document_type}"
                    )

                    # Create directory if needed.
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = f"{output_dir}/{filename_without_extension}.txt"

                with open(str(input_file), "r") as f:
                    orig_text = f.read()

                # Replace the strings.
                replacer = StringReplacer(strings_to_replace, parsed_args.replacement_string)
                scrubbed_text = replacer.replace(orig_text)

                # Write scrubbed text to output file
                try:
                    with open(output_file, "w") as outfile:
                        outfile.write(scrubbed_text)
                    stdout_logger.info(f"Scrubbed file written to {output_file}")
                except OSError as e:
                    stdout_logger.error(f"Error writing scrubbed file: {e}")

                # Save the path to the log.
                output_log.list_replacement_output_file = output_file
                session.add(output_log)

            except Exception as e:
                stdout_logger.error(f"An error occurred while processing {input_file}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args : argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Replace strings in scrubbed text files from a CSV or TSV."
    )
    parser.add_argument(
        "data_file", help="Path to the CSV or TSV file containing the strings."
    )
    parser.add_argument(
        "key_column",
        help="Name of the column in the CSV/TSV containing the keys.",
    )
    parser.add_argument(
        "replacement_string", help="The string to replace the keys with."
    )
    parser.add_argument("output_dir", help="Path to the output directory.")
    parser.add_argument(
        "--document_type",
        type=str,
        default="document",
        help="The document type we want to produce, document, page or block.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="The number of records to process."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="The number of records to skip before beginning processing.",
    )
    parser.add_argument(
        "--debug-sql", action="store_true", help="Enable SQL debugging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        # Fetch file paths from the database.
        results = get_files_to_process(args)
        target_strings = read_target_strings(args.data_file, args.key_column)
        scrub_and_write_files(results, target_strings, args)
    except ValueError as e:
        stdout_logger.error(f"Error reading target strings: {e}")
