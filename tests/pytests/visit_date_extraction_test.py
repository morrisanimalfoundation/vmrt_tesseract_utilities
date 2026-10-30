import argparse
import csv
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from scripts.metadata_miners.visit_date_miner import (
    get_date_pairs_within_days, get_dates_from_tsv, get_values_from_tsv,
    save_visit_dates, set_visit_dates_from_files)
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionMetadata)


@pytest.fixture
def mock_session(mocker):
    """
    Fixture to create a mock SQLAlchemy session.

    Returns
    -------
    mocker.MagicMock
        Mock Session object.
    """
    mock_session = mocker.MagicMock(spec=Session)
    return mock_session


def saved_date_pairs(mock_session):
    """
    Returns the metadata rows passed to the mock session's `add_all`, as sorted tuples.

    Parameters
    ----------
    mock_session : mocker.MagicMock
        Mock SQLAlchemy session.

    Returns
    -------
    list of tuple
        (subject_id, input_id, visit_date, extracted_date) for each row.
    """
    mock_session.add_all.assert_called_once()
    rows = mock_session.add_all.call_args.args[0]
    return sorted((row.subject_id, row.input_id, row.visit_date, row.extracted_date) for row in rows)


@pytest.mark.parametrize(
    'extracted_dates, visit_dates, days, expected_pairs',
    [
        (
            [datetime(2024, 1, 15), datetime(2024, 1, 20)],
            [datetime(2024, 1, 10), datetime(2024, 1, 19)],
            3,
            {(datetime(2024, 1, 19), datetime(2024, 1, 20))},
        ),
        (
            [datetime(2023, 5, 10), datetime(2023, 5, 15)],
            [datetime(2023, 5, 8), datetime(2023, 5, 16)],
            2,
            {  # Use a set instead of a list
                (datetime(2023, 5, 8), datetime(2023, 5, 10)),
                (datetime(2023, 5, 16), datetime(2023, 5, 15)),
            },
        ),
        (
            [datetime(2024, 10, 2)],
            [datetime(2024, 9, 29), datetime(2024, 10, 1)],
            3,
            {  # Use a set instead of a list
                (datetime(2024, 9, 29), datetime(2024, 10, 2)),
                (datetime(2024, 10, 1), datetime(2024, 10, 2)),
            },
        ),
        ([], [], 5, set()),  # Use an empty set
        ([datetime(2024, 7, 4)], [], 10, set()),  # Use an empty set
    ],
)
def test_get_date_pairs_within_days(
    extracted_dates, visit_dates, days, expected_pairs
):
    """
    Test the `get_date_pairs_within_days` function.

    Parameters
    ----------
    extracted_dates : list of datetime
        List of extracted dates.
    visit_dates : list of datetime
        List of visit dates.
    days : int
        Number of days threshold.
    expected_pairs : list of tuple of datetime
        Expected list of date pairs.
    """
    result_pairs = get_date_pairs_within_days(extracted_dates, visit_dates, days)
    assert result_pairs == expected_pairs


def test_set_visit_dates_from_files(mock_session, mocker):
    """
    Test the `set_visit_dates_from_files` function.

    Parameters
    ----------
    mock_session : mocker.MagicMock
        Mock SQLAlchemy session.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    """
    mock_parsed_args = argparse.Namespace(
        visit_date_tsv='test_visit_dates.tsv',
        dog_profile_tsv='test_dog_profile.tsv',
        visit_date_threshold=3,
    )

    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_dates_from_tsv',
        side_effect=[
            [datetime(2024, 1, 10), datetime(2024, 1, 19)],  # visit dates
            [datetime(2024, 1, 1)],  # birthdate
            [],  # death date
        ],
    )
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.DateExtractor.extract_dates_from_file',
        return_value=[datetime(2024, 1, 15), datetime(2024, 1, 20)],
    )

    set_visit_dates_from_files(
        ['test_file.txt'], mock_session, 'test_subject', 1, mock_parsed_args
    )

    # Only 2024-01-20 is within 3 days of a visit (2024-01-19).
    assert saved_date_pairs(mock_session) == [('test_subject', 1, datetime(2024, 1, 19), datetime(2024, 1, 20))]


def test_set_visit_dates_from_files_skips_existing_pairs(mock_session, mocker):
    """
    Test that `set_visit_dates_from_files` doesn't save a date pair the database already has.

    Parameters
    ----------
    mock_session : mocker.MagicMock
        Mock SQLAlchemy session.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    """
    mock_parsed_args = argparse.Namespace(
        visit_date_tsv='test_visit_dates.tsv',
        dog_profile_tsv='test_dog_profile.tsv',
        visit_date_threshold=3,
    )
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_dates_from_tsv',
        side_effect=[
            [datetime(2024, 1, 10), datetime(2024, 1, 19)],  # visit dates
            [datetime(2024, 1, 1)],  # birthdate
            [],  # death date
        ],
    )
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.DateExtractor.extract_dates_from_file',
        return_value=[datetime(2024, 1, 11), datetime(2024, 1, 20)],
    )
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_existing_date_pairs',
        return_value={(datetime(2024, 1, 19), datetime(2024, 1, 20))},
    )

    set_visit_dates_from_files(
        ['test_file.txt'], mock_session, 'test_subject', 1, mock_parsed_args
    )

    assert saved_date_pairs(mock_session) == [('test_subject', 1, datetime(2024, 1, 10), datetime(2024, 1, 11))]


def test_set_visit_dates_from_files_saves_each_pair_once_per_input(mocker, tmp_path):
    """
    Test on a real database that rerunning saves no duplicate date pairs, and that another input can still save them.

    The registration step's empty metadata row should be filled in rather than left behind.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    tmp_path : pathlib.Path
        Temporary directory for the text file.
    """
    text_file = tmp_path / 'record.txt'
    text_file.write_text('Seen on 2024-01-11.\nSeen on 2024-01-20.\n')
    mock_parsed_args = argparse.Namespace(
        visit_date_tsv='test_visit_dates.tsv',
        dog_profile_tsv='test_dog_profile.tsv',
        visit_date_threshold=3,
    )
    dates_by_column = {
        'visit_date': [datetime(2024, 1, 10), datetime(2024, 1, 19)],
        'birth_date': [datetime(2020, 1, 1)],
    }
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_dates_from_tsv',
        side_effect=lambda tsv, subject_id, column: dates_by_column.get(column, []),
    )
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    expected_pairs = [
        (datetime(2024, 1, 10), datetime(2024, 1, 11)),
        (datetime(2024, 1, 19), datetime(2024, 1, 20)),
    ]

    with sessionmaker(bind=engine).begin() as session:
        first_input = TranscriptionInput(input_file='first.pdf', document_type='document')
        second_input = TranscriptionInput(input_file='second.pdf', document_type='document')
        session.add_all([first_input, second_input])
        session.add(TranscriptionMetadata(subject_id='test_subject', transcription_input=first_input))
        session.flush()

        for _ in range(2):
            set_visit_dates_from_files([str(text_file)], session, 'test_subject', first_input.id, mock_parsed_args)
            session.flush()
        set_visit_dates_from_files([str(text_file)], session, 'test_subject', second_input.id, mock_parsed_args)
        session.flush()

        for input_id in (first_input.id, second_input.id):
            rows = session.query(TranscriptionMetadata).filter_by(input_id=input_id).all()
            assert sorted((row.visit_date, row.extracted_date) for row in rows) == expected_pairs


def test_set_visit_dates_from_files_ignores_dates_outside_dogs_life(mock_session, mocker, tmp_path):
    """
    Test that dates in the text from before the dog's birth or after its death are not matched to visits.

    The text file is read for real, so this checks that the birth and death dates reach the date extractor.

    Parameters
    ----------
    mock_session : mocker.MagicMock
        Mock SQLAlchemy session.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    tmp_path : pathlib.Path
        Temporary directory for the text file.
    """
    text_file = tmp_path / 'record.txt'
    text_file.write_text('Seen on 2012-05-01.\nSeen on 2016-05-01.\nSeen on 2020-05-01.\n')
    mock_parsed_args = argparse.Namespace(
        visit_date_tsv='test_visit_dates.tsv',
        dog_profile_tsv='test_dog_profile.tsv',
        visit_date_threshold=3,
    )
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_dates_from_tsv',
        side_effect=[
            [datetime(2012, 5, 1), datetime(2016, 5, 1), datetime(2020, 5, 1)],  # visit dates
            [datetime(2014, 1, 1)],  # birthdate
            [datetime(2019, 1, 1)],  # death date
        ],
    )

    set_visit_dates_from_files(
        [str(text_file)], mock_session, 'test_subject', 1, mock_parsed_args
    )

    assert saved_date_pairs(mock_session) == [('test_subject', 1, datetime(2016, 5, 1), datetime(2016, 5, 1))]


def test_save_visit_dates(mocker):
    """
    Test the `save_visit_dates` function.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    """
    mock_parsed_args = argparse.Namespace(
        debug_sql=False, search_unstructured_text_dir=False, chunk_size=22, offset=0
    )
    mock_session = mocker.MagicMock(spec=Session)
    mock_get_database_session = mocker.patch(
        'scripts.metadata_miners.visit_date_miner.get_database_session',
        return_value=mocker.MagicMock(begin=mocker.MagicMock(return_value=mock_session)),
    )
    mock_session.query().outerjoin().all.return_value = [
        ('test_file.txt', 'test_subject_id')
    ]
    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.set_visit_dates_from_files'
    )

    save_visit_dates(mock_parsed_args)

    mock_get_database_session.assert_called_once_with(echo=False)


@pytest.mark.parametrize(
    'tsv_content, id_value, target_column, expected_values',
    [
        (
            [
                {'grls_id': '123', 'visit_date': '2024-01-10', 'other': 'value1'},
                {'grls_id': '456', 'visit_date': '2023-05-08', 'other': 'value2'},
            ],
            '123',
            'visit_date',
            ['2024-01-10'],
        ),
        (
            [
                {'subject_id': '789', 'birth_date': '2022-10-02', 'other': 'value3'},
                {'subject_id': '101', 'birth_date': '2024-09-29', 'other': 'value4'},
            ],
            '789',
            'birth_date',
            ['2022-10-02'],
        ),
        (
            [
                {'grls_id': '123', 'visit_date': '2024-01-10', 'other': 'value1'},
                {'subject_id': '123', 'visit_date': '2023-05-08', 'other': 'value2'},
            ],
            '123',
            'visit_date',
            ['2024-01-10', '2023-05-08'],
        ),
        ([], '123', 'visit_date', []),
    ],
)
def test_get_values_from_tsv(tsv_content, id_value, target_column, expected_values, tmp_path):
    """
    Test the `get_values_from_tsv` function.

    Parameters
    ----------
    tsv_content : list of dict
        Content of the TSV file.
    id_value : str
        ID value to search for.
    target_column : str
        Name of the column to retrieve values from.
    expected_values : list of str
        Expected list of values.
    tmp_path : pathlib.Path
        Temporary path.
    """
    tsv_file = tmp_path / 'test.tsv'
    with open(tsv_file, 'w', newline='') as f:
        # Explicitly define fieldnames here
        fieldnames = ['grls_id', 'visit_date', 'birth_date', 'other', 'subject_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(tsv_content)

    result = get_values_from_tsv(str(tsv_file), id_value, target_column)
    assert result == expected_values


@pytest.mark.parametrize(
    'tsv_content, grls_id, target_column, expected_dates',
    [
        (
            [
                {'grls_id': '123', 'visit_date': '2024-01-10', 'other': 'value1'},
                {'grls_id': '456', 'visit_date': '2023-05-08', 'other': 'value2'},
            ],
            '123',
            'visit_date',
            [datetime(2024, 1, 10)],
        ),
        (
            [
                {
                    'grls_id': '789',
                    'birth_date': '2022-10-02; 2022-10-03',
                    'other': 'value3',
                },
                {'grls_id': '101', 'birth_date': '2024-09-29', 'other': 'value4'},
            ],
            '789',
            'birth_date',
            [datetime(2022, 10, 2), datetime(2022, 10, 3)],
        ),
        ([], '123', 'visit_date', []),
    ],
)
def test_get_dates_from_tsv(
    tsv_content, grls_id, target_column, expected_dates, tmp_path, mocker
):
    """
    Test the `get_dates_from_tsv` function.

    Parameters
    ----------
    tsv_content : list of dict
        Content of the TSV file.
    grls_id : str
        GRLS ID value to search for.
    target_column : str
        Name of the column to retrieve dates from.
    expected_dates : list of datetime
        Expected list of dates.
    tmp_path : pathlib.Path
        Temporary path.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture.
    """
    tsv_file = tmp_path / 'test.tsv'
    with open(tsv_file, 'w', newline='') as f:
        if tsv_content:
            writer = csv.DictWriter(f, tsv_content[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(tsv_content)
        else:
            pass

    mocker.patch(
        'scripts.metadata_miners.visit_date_miner.find_dates',
        side_effect=lambda date_str: [
            datetime.strptime(d.strip(), '%Y-%m-%d') for d in date_str.split(';')
        ],
    )

    result = get_dates_from_tsv(str(tsv_file), grls_id, target_column)
    assert result == expected_dates
