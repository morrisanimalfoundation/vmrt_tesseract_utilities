"""
Tests that every pipeline step stops with the SQL that fixes it, rather than MySQL's "Unknown column" error, on a
database created before the text_source column was added (issue #82).

Each step is run as a script, the way the README runs it, against a SQLite database whose transcription_output table
has no text_source column.
"""

import importlib.util
import os
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, text

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.fixture
def old_database(tmp_path):
    """
    A database made before #82: its transcription_output table has every column but text_source.
    """
    url = f'sqlite:///{tmp_path}/old.db'
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE transcription_input (id INTEGER PRIMARY KEY, document_type VARCHAR(30), '
                                'created DATETIME, input_file VARCHAR(512), ocr_status VARCHAR(20), ocr_error TEXT)'))
        connection.execute(text('CREATE TABLE transcription_output (id INTEGER PRIMARY KEY, input_id INTEGER, '
                                'ocr_output_file VARCHAR(512), ocr_confidence FLOAT, '
                                'list_replacement_output_file VARCHAR(512), pii_scrubber_output_file VARCHAR(512), '
                                'pii_scrubber_confidence_file VARCHAR(512), pii_scrubber_status VARCHAR(20), '
                                'pii_scrubber_error TEXT)'))
    tsv = tmp_path / 'dog_profile.tsv'
    tsv.write_text('subject_id\tcall_name\tbirth_date\tdeath_date\n094-000001\tBiscuit\t2014-05-02\t\n')
    visits = tmp_path / 'vet_visits.tsv'
    visits.write_text('subject_id\tvisit_date\n094-000001\t2023-03-14\n')
    return url, str(tsv), str(visits), str(tmp_path / 'output')


@pytest.mark.parametrize('step', ['replace', 'scrub', 'mine'])
def test_steps_stop_with_the_sql_to_add_the_column(old_database, step):
    url, tsv, visits, output = old_database
    if step == 'scrub' and not _has('presidio_analyzer'):
        pytest.skip('The PII scrubber packages are only installed in the workspace image.')
    command = {
        'replace': ['scripts/replace_strings.py', tsv, 'subject_id', '<ID>', output],
        'scrub': ['scripts/scrubbers/pii_scrubber.py', output],
        'mine': ['scripts/metadata_miners/visit_date_miner.py', output, f'--visit-date-tsv={visits}',
                 f'--dog-profile-tsv={tsv}'],
    }[step]
    env = {**os.environ, 'SQL_CONNECTION_STRING': url, 'PYTHONPATH': REPO}
    done = subprocess.run([sys.executable, *command], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 1, done.stdout + done.stderr
    log = done.stdout + done.stderr
    assert 'ALTER TABLE transcription_output ADD COLUMN text_source VARCHAR(20);' in log
    assert 'Unknown column' not in log and 'no such column' not in log
