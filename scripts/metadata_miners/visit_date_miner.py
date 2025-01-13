import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Tuple

import sqlalchemy
from sqlalchemy import and_, tuple_

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
) -> List[Tuple[datetime, datetime]]:
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
    List[Tuple[datetime, datetime]]
        A list of unique (visit_date, extracted_date) pairs.
    """

    result_pairs = set()
    for visit_date in visit_dates:
        for extracted_date in extracted_dates:
            if abs(extracted_date - visit_date) <= timedelta(days=days):
                result_pairs.add((visit_date, extracted_date))
                break  # Move to the next visit_date once a match is found

    return list(result_pairs)


def get_values_from_table(session, table_class, **kwargs) -> list:
    """
    Retrieves values from a specified table based on optional filter conditions.

    This function allows you to query a database table and retrieve specific columns
    or filter the results based on provided criteria. It uses SQLAlchemy's ORM for
    database interaction.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        The SQLAlchemy session to use for the query.
    table_class : sqlalchemy.ext.declarative.api.DeclarativeMeta
        The SQLAlchemy model class representing the table.
    **kwargs
        Optional keyword arguments:
            target_columns (list): A list of column names to retrieve.
                                    If not provided, all columns are retrieved.
            unique (bool): If True, only unique rows will be returned. Defaults to False.
            Other keyword arguments specifying filter conditions in the form
            `column_name=column_value`.

    Returns
    -------
    list
        A list of table objects or tuples (if target_columns is specified) matching the criteria.

    """

    try:
        target_columns = kwargs.pop("target_columns", None)
        unique = kwargs.pop("unique", False)  # Extract unique flag
        query = session.query(table_class)

        if kwargs:
            filter_conditions = [
                getattr(table_class, col) == val for col, val in kwargs.items()
            ]
            query = query.filter(and_(*filter_conditions))

        if target_columns:
            columns_to_fetch = [getattr(table_class, col) for col in target_columns]
            query = query.with_entities(*columns_to_fetch)

        if unique:
            query = query.distinct()

        results = query.all()
        return results
    except Exception as e:
        stdout_logger.error(
            f"Error retrieving records from {table_class.__tablename__}: {e}"
        )
        return []


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
        parsed_args.visit_date_tsv, subject_id, "visit_date"
    )
    dogs_birth_dates = (
        get_dates_from_tsv(parsed_args.dog_profile_tsv, subject_id, "birth_date")
        if parsed_args.dog_profile_tsv
        else None
    )
    dogs_birth_date = dogs_birth_dates[0] if dogs_birth_dates else None
    dogs_death_dates = (
        get_dates_from_tsv(parsed_args.dog_profile_tsv, subject_id, "death_date")
        if parsed_args.dog_profile_tsv
        else None
    )
    dogs_death_date = dogs_death_dates[0] if dogs_death_dates else None

    extracted_dates = []
    for file in files:
        extractor = DateExtractor(file, dogs_birth_date, dogs_death_date)
        extracted_dates.extend(extractor.extract_dates_from_file())

    if extracted_dates:
        date_pairs = get_date_pairs_within_days(
            extracted_dates, dogs_visit_dates, parsed_args.visit_date_threshold
        )

        # Check for existing records in bulk (example using a WHERE IN clause)
        existing_records = (
            session.query(TranscriptionMetadata)
            .filter(
                and_(
                    TranscriptionMetadata.subject_id == subject_id,
                    TranscriptionMetadata.input_id == input_id,
                    tuple_(
                        TranscriptionMetadata.visit_date,
                        TranscriptionMetadata.extracted_date,
                    ).in_(date_pairs),
                )
            )
            .all()
        )

        # Create new records only for date pairs that don't exist
        existing_date_pairs = {
            (record.visit_date, record.extracted_date) for record in existing_records
        }
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

        session.add_all(new_metadata)  # Batch insert


def save_visit_dates(parsed_args: argparse.Namespace) -> List[datetime]:
    """
    Gets a list of dates from the output files. Optimized to reduce database interactions.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed args.

    Returns
    -------
    List[date]
        The list of visit dates extracted from the files.
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
            .all()
        )

        for row, subject_id in results:
            files_to_check = [row.pii_scrubber_confidence_file]
            if parsed_args.search_unstructured_text_dir:
                files_to_check.append(row.ocr_output_file)
            set_visit_dates_from_files(
                files_to_check, session, subject_id, row.input_id, parsed_args
            )

    return []  # The function doesn't actually use extracted_dates


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
        with open(tsv_file, "r", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                if "grls_id" in row and row["grls_id"] == id_value:
                    values.append(row[target_column])
                elif "subject_id" in row and str(row["subject_id"]) == id_value:
                    values.append(row[target_column])
    except Exception as e:
        stdout_logger.error(f"Error getting values from TSV: {e}")

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
        stdout_logger.error(f"Error extracting dates for GRLS ID {grls_id}: {e}")

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
        description="Search files for metadata values."
    )
    arg_parser.add_argument(
        "output_dir", type=str, help="Path to directory containing the output files."
    )
    arg_parser.add_argument(
        "--visit_date_tsv",
        required=True,
        type=str,
        help="Path to a TSV file containing the vet visit dates keyed by visit_date and grls_id.",
    )
    arg_parser.add_argument(
        "--dog_profile_tsv", type=str, help="Path to a TSV file containing the dog profile data."
    )
    arg_parser.add_argument(
        "--search_unstructured_text_dir",
        action="store_true",
        required=False,
        help="Search unstructured text files",
    )
    arg_parser.add_argument("--debug-sql", action="store_true", help="Enable SQL debugging")
    arg_parser.add_argument(
        "--visit_date_threshold",
        type=int,
        default=3,
        help="The number of days to consider as the threshold relative to the visit dates.",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_visit_dates(args)