"""
End-to-end test of the whole pipeline against a real MySQL database.

Each pipeline step reads the previous step's results from the database, so package updates (SQLAlchemy, PyMySQL,
pdf2image, poppler, Tesseract, presidio) can break the hand-off between steps without any single step failing on its
own. This script generates synthetic records for made-up dogs, runs the five steps from the README's "Running the
pipeline" section exactly as a user would, in both document and page mode, and then checks the database and the
files the steps wrote.

Run it inside the workspace image, with SQL_CONNECTION_STRING pointing at an empty MySQL database, for example:

    docker network create vmrt-pipeline-test
    docker run -d --name vmrt-pipeline-mysql --network vmrt-pipeline-test -e MYSQL_ROOT_PASSWORD=test \\
        -e MYSQL_DATABASE=vmrt_emr_transcription mysql:8.0 --default-authentication-plugin=mysql_native_password
    docker run --rm --user 1234:1234 --network vmrt-pipeline-test -v "$PWD:/workspace" \\
        -e SQL_CONNECTION_STRING=mysql+pymysql://root:test@vmrt-pipeline-mysql:3306/vmrt_emr_transcription \\
        <image> python tests/pipeline_test.py

It refuses to run against a database that already has pipeline data in it. It exits with 0 if everything worked and
1 otherwise, listing every problem it found. All of the data is made up.
"""

import csv
import datetime
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import sqlalchemy

from tests.synthetic_pdfs import SyntheticPage, write_pdf
from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionMetadata,
                                               TranscriptionOutput,
                                               get_database_session)

DOCUMENT_TYPES = ('document', 'page')
DB_WAIT_SECONDS = 120
# The placeholders the two string replacement passes put in.
ID_PLACEHOLDER = '<ID>'
NAME_PLACEHOLDER = '<NAME>'


@dataclass
class Record:
    """
    One synthetic medical record PDF.

    Attributes
    ----------
    filename : str
        The PDF's name inside its dog's folder.
    pages : List[SyntheticPage]
        The pages.
    visit_dates : Set[Tuple[str, str]]
        The (visit date from vet_visits.tsv, date written in the record) pairs the miner should find, as ISO dates.
    """
    filename: str
    pages: List[SyntheticPage]
    visit_dates: Set[Tuple[str, str]]


@dataclass
class Dog:
    """
    A made-up study dog, with its profile, known visits and records.
    """
    subject_id: str
    call_name: str
    birth_date: str
    visit_dates: List[str]
    records: List[Record] = field(default_factory=list)


def record_header(dog: Dog, owner: str, phone: str) -> List[str]:
    """
    The first lines of a record: clinic, patient and owner, the identifiers later steps should remove.
    """
    return [
        'Sunny Meadows Animal Hospital',
        f'Patient: {dog.call_name}   GRLS ID: {dog.subject_id}',
        f'Owner: {owner}   Phone: {phone}',
    ]


def make_dogs() -> List[Dog]:
    """
    The dogs and their records. Each record has one visit date that matches vet_visits.tsv (one of them a day off,
    within the miner's threshold), and some also have a date that matches no visit, which must not be stored.
    """
    biscuit = Dog('094-000001', 'Biscuit', '2014-05-02', ['2023-03-14', '2024-03-20', '2025-01-10'])
    juniper = Dog('094-000002', 'Juniper', '2016-08-19', ['2023-07-02'])
    biscuit.records = [
        Record('2023-03-14 wellness.pdf', [
            SyntheticPage(record_header(biscuit, 'Margaret Thompson', '541-555-0163') + [
                'Visit date: March 14, 2023',
                'Exam: Weight 31.2 kg. Bright, alert and responsive.',
            ], font_size=31),
            SyntheticPage([
                'Plan: Rabies and DHPP vaccines given.',
                'Next appointment: September 14, 2023 with Dr. Alan Whitaker.',
            ], font_size=31),
        ], {('2023-03-14', '2023-03-14')}),
        Record('2024-03-20 recheck.pdf', [
            SyntheticPage(record_header(biscuit, 'Margaret Thompson', '541-555-0163') + [
                'Visit date: March 21, 2024',
                'Recheck of dental tartar. Mild improvement.',
            ], font_size=31),
        ], {('2024-03-20', '2024-03-21')}),
    ]
    juniper.records = [
        Record('2023-07-02 exam.pdf', [
            SyntheticPage(record_header(juniper, 'Daniel Okafor', '503-555-0118') + [
                'Visit date: July 2, 2023',
                'History: Limping on the left hind leg for two days.',
            ], font_size=31),
            SyntheticPage([
                'Radiographs of the left stifle: no fracture seen.',
                'Plan: Rest and recheck if not improved.',
            ], font_size=31),
            # Scanned sideways, as some real pages are.
            SyntheticPage([
                'Discharge: carprofen 75 mg by mouth twice daily.',
            ], font_size=31, rotation=90),
        ], {('2023-07-02', '2023-07-02')}),
    ]
    return [biscuit, juniper]


# Identifiers the scrubber (step 4) should remove from the text. Subject IDs and dog names are removed by step 3.
SCRUBBED_IDENTIFIERS = ['Margaret Thompson', 'Daniel Okafor', '555-0163', '555-0118', 'Alan Whitaker']


def write_inputs(root: str, dogs: List[Dog]) -> Tuple[str, str, str]:
    """
    Writes the dogs' records into folders named after their subject IDs, as the real records are, and the two TSVs.

    Returns
    -------
    tuple
        The records folder, dog_profile.tsv and vet_visits.tsv paths.
    """
    data_dir = os.path.join(root, 'data')
    for dog in dogs:
        dog_dir = os.path.join(data_dir, dog.subject_id)
        os.makedirs(dog_dir)
        for record in dog.records:
            write_pdf(os.path.join(dog_dir, record.filename), record.pages)
    profile_tsv = os.path.join(root, 'dog_profile.tsv')
    with open(profile_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['subject_id', 'call_name', 'birth_date', 'death_date'])
        for dog in dogs:
            writer.writerow([dog.subject_id, dog.call_name, dog.birth_date, ''])
    visits_tsv = os.path.join(root, 'vet_visits.tsv')
    with open(visits_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['subject_id', 'visit_date'])
        for dog in dogs:
            for visit_date in dog.visit_dates:
                writer.writerow([dog.subject_id, visit_date])
    return data_dir, profile_tsv, visits_tsv


def wait_for_database() -> None:
    """
    Waits until MySQL accepts connections, since a freshly started server takes a while to initialise.
    """
    engine = get_database_session().kw['bind']
    deadline = time.monotonic() + DB_WAIT_SECONDS
    while True:
        try:
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text('SELECT 1'))
            return
        except sqlalchemy.exc.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def check_database_is_empty() -> None:
    """
    Stops the test if the database already has pipeline data, so it can't mix with or be mistaken for real data.
    """
    engine = get_database_session().kw['bind']
    if not sqlalchemy.inspect(engine).has_table(TranscriptionInput.__tablename__):
        return
    with engine.connect() as connection:
        count = connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(
            TranscriptionInput.__table__)).scalar()
    if count:
        sys.exit(f'The database already has {count} transcription_input rows. Run this test against an empty '
                 f'database only.')


def run(*command: str) -> None:
    """
    Runs one pipeline step the way the README does, and stops the test if it fails.
    """
    print(f'\n$ {" ".join(command)}', flush=True)
    subprocess.run(['python', *command], check=True)


def run_pipeline(document_type: str, data_dir: str, output_dir: str, profile_tsv: str, visits_tsv: str) -> None:
    """
    Runs steps 1 to 5 from the README in one document mode.
    """
    mode = f'--document-type={document_type}'
    run('scripts/create_transcription_process.py', data_dir, mode)
    run('scripts/transcribe_pdfs.py', output_dir, mode)
    # Two replacement passes, as the README suggests: subject IDs, then dog names.
    run('scripts/replace_strings.py', profile_tsv, 'subject_id', ID_PLACEHOLDER, output_dir, mode, '--until-done')
    run('scripts/replace_strings.py', profile_tsv, 'call_name', NAME_PLACEHOLDER, output_dir, mode, '--until-done')
    run('scripts/scrubbers/pii_scrubber.py', output_dir, mode, '--until-done')
    run('scripts/metadata_miners/visit_date_miner.py', output_dir, f'--visit-date-tsv={visits_tsv}',
        f'--dog-profile-tsv={profile_tsv}')


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def check_results(document_type: str, dogs: List[Dog], data_dir: str) -> List[str]:
    """
    Checks what one document mode left in the database and on disk.

    Returns
    -------
    list of str
        The problems found; empty if there are none.
    """
    problems = []
    records = {os.path.join(data_dir, dog.subject_id, record.filename): (dog, record)
               for dog in dogs for record in dog.records}
    with get_database_session()() as session:
        inputs = session.query(TranscriptionInput).filter(TranscriptionInput.document_type == document_type).all()
        if sorted(i.input_file for i in inputs) != sorted(records):
            problems.append(f'{document_type}: registered {sorted(i.input_file for i in inputs)}, '
                            f'expected {sorted(records)}')
        for input_row in inputs:
            dog, record = records.get(input_row.input_file, (None, None))
            if record is None:
                continue
            name = f'{document_type} {dog.subject_id}/{record.filename}'
            if input_row.ocr_status != 'success':
                problems.append(f'{name}: ocr_status is {input_row.ocr_status!r} ({input_row.ocr_error})')

            # One output per file in document mode, one per page in page mode.
            outputs = session.query(TranscriptionOutput).filter(TranscriptionOutput.input_id == input_row.id).all()
            expected_outputs = 1 if document_type == 'document' else len(record.pages)
            if len(outputs) != expected_outputs:
                problems.append(f'{name}: {len(outputs)} outputs, expected {expected_outputs}')
            for output in outputs:
                problems.extend(check_output(f'{name} output {output.id}', output, dog))

            # The visit dates the miner matched, ignoring the registration row that has none.
            found = {(m.visit_date.date().isoformat(), m.extracted_date.date().isoformat())
                     for m in session.query(TranscriptionMetadata).filter(
                         TranscriptionMetadata.input_id == input_row.id,
                         TranscriptionMetadata.visit_date.isnot(None)).all()}
            if found != record.visit_dates:
                problems.append(f'{name}: visit dates (visit, extracted) {sorted(found)}, '
                                f'expected {sorted(record.visit_dates)}')
            subject_ids = {m.subject_id for m in input_row.derived_metadata}
            if subject_ids != {dog.subject_id}:
                problems.append(f'{name}: metadata subject IDs {subject_ids}, expected {dog.subject_id}')
    return problems


def check_output(name: str, output: TranscriptionOutput, dog: Dog) -> List[str]:
    """
    Checks one transcription_output row: every step filled in its column, the files exist, and the identifiers each
    step should remove are gone.
    """
    problems = []
    columns = ('ocr_output_file', 'list_replacement_output_file', 'pii_scrubber_output_file',
               'pii_scrubber_confidence_file')
    for column in columns:
        path = getattr(output, column)
        if not path:
            problems.append(f'{name}: {column} is empty')
        elif not os.path.isfile(path):
            problems.append(f'{name}: {column} {path} does not exist')
    if problems:
        return problems

    replaced = read_file(output.list_replacement_output_file)
    for identifier in (dog.subject_id, dog.call_name):
        if identifier.lower() in replaced.lower():
            problems.append(f'{name}: {identifier!r} is still in the string replacement output')
    scrubbed = read_file(output.pii_scrubber_output_file)
    for identifier in [dog.subject_id, dog.call_name] + SCRUBBED_IDENTIFIERS:
        if identifier.lower() in scrubbed.lower():
            problems.append(f'{name}: {identifier!r} is still in the scrubbed output')
    return problems


def main() -> int:
    if not os.environ.get('SQL_CONNECTION_STRING'):
        sys.exit('Set SQL_CONNECTION_STRING to an empty MySQL database (see the docstring at the top of this file).')
    # The steps are run from the repository folder, as the README does.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wait_for_database()
    check_database_is_empty()
    dogs = make_dogs()
    started = datetime.datetime.now()
    with tempfile.TemporaryDirectory() as root:
        data_dir, profile_tsv, visits_tsv = write_inputs(root, dogs)
        output_dir = os.path.join(root, 'output')
        run('scripts/database_setup.py', 'install')
        problems: List[str] = []
        for document_type in DOCUMENT_TYPES:
            run_pipeline(document_type, data_dir, output_dir, profile_tsv, visits_tsv)
            problems.extend(check_results(document_type, dogs, data_dir))
    elapsed = (datetime.datetime.now() - started).total_seconds()
    if problems:
        print(f'\nPipeline test FAILED after {elapsed:.0f} seconds:', *problems, sep='\n- ')
        return 1
    print(f'\nPipeline test passed in {elapsed:.0f} seconds ({", ".join(DOCUMENT_TYPES)} modes).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
