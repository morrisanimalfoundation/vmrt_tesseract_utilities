import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import sqlalchemy

from vmrt_tesseract_utilities.database import (TranscriptionMetadata,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.date_extractor import DateExtractor, find_dates
from vmrt_tesseract_utilities.logging import stdout_logger

"""
Extracts visit date metadata from files and stashes them in the metadata database.
"""


def get_date_pairs_within_days(
        extracted_dates: List[datetime], visit_dates: List[datetime], days: int
) -> Set[Tuple[datetime, datetime]]:
    """
    Finds pairs of dates (visit_date, extracted_date) where the extracted_date is within
    a specified number of days of the visit_date, without duplicate pairs. Optimized
    to reduce redundant iterations.

    Parameters
    ----------
    extracted_dates : List[datetime]
        The list of dates to search within.
    visit_dates : List[datetime]
        The list of dates to compare against.
    days : int
        The number of days to consider as the threshold.

    Returns
    -------
    Set[Tuple[datetime, datetime]]
        A set of unique (visit_date, extracted_date) pairs.
    """

    result_pairs = set()
    for visit_date in visit_dates:
        for extracted_date in extracted_dates:
            if abs(extracted_date - visit_date) <= timedelta(days=days):
                result_pairs.add((visit_date, extracted_date))
                break  # Move to the next visit_date once a match is found

    return result_pairs


def get_dog_dates(parsed_args: argparse.Namespace, subject_id: str) -> Tuple[
    Optional[datetime.date], Optional[datetime.date]]:
    """
    Retrieves the dog's birth and death dates from the TSV files.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed arguments.
    subject_id : str
        The subject ID.

    Returns
    -------
    tuple of Optional[datetime.date]
        A tuple containing the dog's birthdate and death date.
        Each date can be None if not found.
    """
    dogs_birth_dates = (
        get_dates_from_tsv(parsed_args.dog_profile_tsv, subject_id, 'birth_date')
        if parsed_args.dog_profile_tsv
        else None
    )
    dogs_birth_date = dogs_birth_dates[0] if dogs_birth_dates else None
    dogs_death_dates = (
        get_dates_from_tsv(parsed_args.dog_profile_tsv, subject_id, 'death_date')
        if parsed_args.dog_profile_tsv
        else None
    )
    dogs_death_date = dogs_death_dates[0] if dogs_death_dates else None
    return dogs_birth_date, dogs_death_date


def extract_dates_from_files(files: list, dogs_birth_date: Optional[datetime.date],
                             dogs_death_date: Optional[datetime.date]) -> List[datetime.date]:
    """
    Extracts dates from the given list of files.

    Parameters
    ----------
    files : list
        A list of filepaths to the files to extract from.
    dogs_birth_date : Optional[datetime.date]
        The dog's birthdate.
    dogs_death_date : Optional[datetime.date]
        The dog's death date.

    Returns
    -------
    list of datetime.date
        A list of extracted dates.
    """
    extracted_dates = []
    for file in files:
        extractor = DateExtractor(file, dogs_birth_date, dogs_death_date)
        extracted_dates.extend(extractor.extract_dates_from_file())
    return extracted_dates


def update_existing_records(session: sqlalchemy.orm.session.Session, subject_id: str, input_id: int,
                            date_pairs: List[Tuple[datetime.date, datetime.date]]) -> None:
    """
    Updates existing records with NULL visit_date and extracted_date.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        The SQLAlchemy session to use for the query.
    subject_id : str
        The subject ID.
    input_id : int
        The record's input_id.
    date_pairs : list of tuple of (datetime.date, datetime.date)
        A list of date pairs (visit_date, extracted_date).
    """
    # ... (rest of the function code)

    for visit_date, extracted_date in date_pairs:
        session.query(TranscriptionMetadata).filter(
            TranscriptionMetadata.subject_id == subject_id,
            TranscriptionMetadata.input_id == input_id,
            TranscriptionMetadata.visit_date.is_(None),
            TranscriptionMetadata.extracted_date.is_(None),
        ).update({'visit_date': visit_date, 'extracted_date': extracted_date})


def get_existing_date_pairs(session: sqlalchemy.orm.session.Session, subject_id: str, input_id: int,
                            date_pairs: set[tuple[datetime, datetime]]) -> Set[
    Tuple[datetime.date, datetime.date]]:
    """
    Retrieves existing date pairs from the database.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        The SQLAlchemy session to use for the query.
    subject_id : str
        The subject ID.
    input_id : int
        The record's input_id.
    date_pairs : set[tuple[datetime, datetime]]
        A set of date pairs (visit_date, extracted_date).

    Returns
    -------
    set of tuple of (datetime.date, datetime.date)
        A set of existing date pairs.
    """
    existing_records = (
        session.query(TranscriptionMetadata)
        .filter(
            sqlalchemy.and_(
                TranscriptionMetadata.subject_id == subject_id,
                TranscriptionMetadata.input_id == input_id,
                sqlalchemy.tuple_(
                    TranscriptionMetadata.visit_date,
                    TranscriptionMetadata.extracted_date,
                ).in_(date_pairs),
            )
        )
        .all()
    )
    return {(record.visit_date, record.extracted_date) for record in existing_records}


def set_visit_dates_from_files(
        files: list,
        session: sqlalchemy.orm.session.Session,
        subject_id: str,
        input_id: int,
        parsed_args: argparse.Namespace,
):
    """
    Gets a list of dates from the specified files and adds them to the database
    if they don't already exist. Optimized to reduce database interactions.

    Parameters
    ----------
    files : list
        A list of filepaths to the files to extract from.
    session : sqlalchemy.orm.session.Session
        The SQLAlchemy session to use for the query.
    subject_id : str
        The subject_id of the record.
    input_id : int
        The record's input_id.
    parsed_args : argparse.Namespace
        The parsed args.
    """

    dogs_visit_dates = get_dates_from_tsv(
        parsed_args.visit_date_tsv, subject_id, 'visit_date'
    )
    dogs_birth_date, dogs_death_date = get_dog_dates(parsed_args, subject_id)

    extracted_dates = extract_dates_from_files(files, dogs_birth_date, dogs_death_date)

    if extracted_dates:
        date_pairs = get_date_pairs_within_days(
            extracted_dates, dogs_visit_dates, parsed_args.visit_date_threshold
        )

        update_existing_records(session, subject_id, input_id, date_pairs)

        existing_date_pairs = get_existing_date_pairs(session, subject_id, input_id, date_pairs)

        new_metadata = [
            TranscriptionMetadata(
                subject_id=subject_id,
                input_id=input_id,
                extracted_date=extracted_date,
                visit_date=visit_date,
            )
            for visit_date, extracted_date in date_pairs
            if (visit_date, extracted_date) not in existing_date_pairs
        ]
        session.add_all(new_metadata)


def save_visit_dates(parsed_args: argparse.Namespace):
    """
    Gets a list of dates from the output files. Optimized to reduce database interactions.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed args.
    """

    session_maker = get_database_session(echo=parsed_args.debug_sql)
    with session_maker.begin() as session:
        # Fetch all necessary data from TranscriptionOutput and TranscriptionMetadata in one query
        results = (
            session.query(TranscriptionOutput, TranscriptionMetadata.subject_id)
            .outerjoin(
                TranscriptionMetadata,
                TranscriptionOutput.input_id == TranscriptionMetadata.input_id,
            )
            # Filter for records where visit_date is NULL
            .filter(TranscriptionMetadata.visit_date.is_(None))
            # Apply offset and limit for chunking
            .offset(parsed_args.offset)
            .limit(parsed_args.chunk_size)
            .all()
        )

        stdout_logger.info(f'Searching {len(results)} files for visit dates.')
        for row, subject_id in results:
            files_to_check = [row.pii_scrubber_confidence_file]
            if parsed_args.search_unstructured_text_dir:
                files_to_check.append(row.ocr_output_file)
            set_visit_dates_from_files(
                files_to_check, session, subject_id, row.input_id, parsed_args
            )
        stdout_logger.info(f'Finished searching {len(results)} files for visit dates.')


def get_values_from_tsv(tsv_file, id_value, target_column):
    """
    Retrieves the values of a specified column from a TSV file where either
    the `grls_id` or `subject_id` column matches a given value, using csv.DictReader.
    Prioritizes `grls_id` if both exist.

    Parameters
    ----------
    tsv_file : str
        Path to the TSV file.
    id_value : str
        The ID value to search for (will check `grls_id` first, then `subject_id`).
    target_column : str
        The name of the column whose values you want to retrieve.

    Returns
    -------
    list
        A list of values from the `target_column` in the matching rows.
    """

    values = []
    try:
        with open(tsv_file, 'r', newline='') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                if 'grls_id' in row and row['grls_id'] == id_value:
                    values.append(row[target_column])
                elif 'subject_id' in row and str(row['subject_id']) == id_value:
                    values.append(row[target_column])
    except Exception as e:
        stdout_logger.error(f'Error getting values from TSV: {e}')

    return values


def get_dates_from_tsv(tsv_file: str, grls_id: str, target_column: str) -> List[datetime]:
    """
    Extracts `visit_date` values from a TSV file where the `grls_id`
    matches the given value.

    Parameters
    ----------
    tsv_file : str
        Path to the TSV file.
    grls_id : str
        The GRLS ID of the record.
    target_column : str
        The name of the column whose values you want to retrieve

    Returns
    -------
    List[datetime]
        A list of a dog's visit dates.
    """

    extracted_data = []
    try:
        date_strings = get_values_from_tsv(tsv_file, grls_id, target_column)
        for date_str in date_strings:
            extracted_data.extend(find_dates(date_str))
    except Exception as e:
        stdout_logger.error(f'Error extracting dates for GRLS ID {grls_id}: {e}')

    return extracted_data


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    arg_parser = argparse.ArgumentParser(
        description='Search files for metadata values.'
    )
    arg_parser.add_argument(
        'output_dir', type=str, help='Path to directory containing the output files.'
    )
    arg_parser.add_argument(
        '--visit_date_tsv',
        required=True,
        type=str,
        help='Path to a TSV file containing the vet visit dates keyed by visit_date and grls_id.',
    )
    arg_parser.add_argument(
        '--dog_profile_tsv', required=True, type=str, help='Path to a TSV file containing the dog profile data.'
    )
    arg_parser.add_argument(
        '--search_unstructured_text_dir',
        action='store_true',
        help='Search unstructured text files',
    )
    arg_parser.add_argument('--debug_sql', action='store_true', help='Enable SQL debugging')
    arg_parser.add_argument(
        '--visit_date_threshold',
        type=int,
        default=3,
        help='The number of days to consider as the threshold relative to the visit dates.',
    )
    arg_parser.add_argument('--chunk_size', type=int, help='The number of records to process.')
    arg_parser.add_argument('--offset', type=int, default=0,
                            help='The number of records to skip before beginning processing.')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_visit_dates(args)
