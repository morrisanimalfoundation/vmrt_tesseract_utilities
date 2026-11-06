"""
Tests for how the `pii_scrubber` script picks files, records failures and splits long texts.

These use a stand-in for the analyzer, so they need Presidio's packages but not the model. The packages are only
installed in the workspace image, so the tests are skipped elsewhere.
"""

import argparse
import logging
import re
import types

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

for module in ('presidio_analyzer', 'presidio_anonymizer'):
    pytest.importorskip(module, reason='The PII scrubber packages are only installed in the workspace image.')

from presidio_analyzer import RecognizerResult  # noqa: E402

from scripts.scrubbers import pii_scrubber  # noqa: E402
from vmrt_tesseract_utilities import database  # noqa: E402


class FakeAnalyzer:
    """
    Finds each "Secret" in the text as a PERSON, and fails on text containing "BROKEN".

    Records the length of every text it's given.
    """

    def __init__(self):
        self.lengths = []

    def analyze(self, text, language):
        self.lengths.append(len(text))
        if 'BROKEN' in text:
            raise ValueError('[E088] Text of length 1165500 exceeds maximum of 1000000.')
        return [RecognizerResult('PERSON', match.start(), match.end(), 0.9) for match in re.finditer('Secret', text)]


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    A SQLite database with three OCR'd files waiting to be scrubbed, the second of which makes the analyzer fail.
    """
    db_url = f'sqlite:///{tmp_path}/test.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    database.Base.metadata.create_all(engine)
    with Session(engine) as session:
        for name, content in (('good', 'Seen by Dr. Secret.\n'), ('bad', 'BROKEN\n'), ('other', 'No names.\n')):
            path = tmp_path / f'{name}.txt'
            path.write_text(content)
            session.add(database.TranscriptionInput(document_type='document', input_file=f'{name}.pdf',
                                                    assets=[database.TranscriptionOutput(ocr_output_file=str(path))]))
        session.commit()
    return engine


def _run(tmp_path, analyzer, retry_failed=False, until_done=True, chunk_size=10):
    args = argparse.Namespace(output_to=str(tmp_path / 'output'), document_type='document', chunk_size=chunk_size,
                              offset=0, debug_sql=False, threshold=0.0, until_done=until_done,
                              retry_failed=retry_failed)
    return pii_scrubber.scrub_chunks(args, analyzer)


def _statuses(engine) -> dict:
    with Session(engine) as session:
        return {o.transcription_input.input_file: (o.pii_scrubber_status, o.pii_scrubber_error,
                                                   o.pii_scrubber_output_file is not None)
                for o in session.query(database.TranscriptionOutput)}


def test_failure_is_recorded_and_the_rest_are_scrubbed(tmp_path, engine, caplog):
    """
    Test that a file the analyzer fails on is saved as failed with its error, and doesn't stop the other files.
    """
    with caplog.at_level(logging.INFO):
        failures = _run(tmp_path, FakeAnalyzer(), chunk_size=1)

    error = 'ValueError: [E088] Text of length 1165500 exceeds maximum of 1000000.'
    assert [error for _, error in failures] == [error]
    assert _statuses(engine) == {
        'good.pdf': ('success', None, True),
        'bad.pdf': ('failed', error, False),
        'other.pdf': ('success', None, True),
    }
    assert 'Scrubbed 2 of 3 files.' in caplog.text
    assert f'bad.txt: {error}' in caplog.text
    scrubbed = (tmp_path / 'output/scrubbed_text/document/scrubbed_document/good.txt').read_text()
    assert scrubbed == 'Seen by Dr. <PERSON>.\n'


def test_failed_files_are_only_retried_when_asked(tmp_path, engine):
    """
    Test that a failed file is skipped on the next run, and scrubbed with --retry-failed, which clears its error.
    """
    _run(tmp_path, FakeAnalyzer())

    analyzer = FakeAnalyzer()
    assert _run(tmp_path, analyzer) == []
    assert analyzer.lengths == []

    (tmp_path / 'bad.txt').write_text('Fixed.\n')
    assert _run(tmp_path, analyzer, retry_failed=True) == []
    assert _statuses(engine)['bad.pdf'] == ('success', None, True)


@pytest.mark.parametrize('chunk_size', [1, 10])
def test_retry_failed_tries_a_failing_file_once_per_run(tmp_path, engine, chunk_size):
    """
    Test that with --retry-failed and --until-done, a file that fails again isn't picked over and over.
    """
    _run(tmp_path, FakeAnalyzer())

    analyzer = FakeAnalyzer()
    failures = _run(tmp_path, analyzer, retry_failed=True, chunk_size=chunk_size)

    assert len(failures) == 1
    assert analyzer.lengths == [len('BROKEN\n')]


@pytest.mark.parametrize('until_done, chunks_scrubbed', [(False, 1), (True, 2)])
def test_scrub_chunks(monkeypatch, until_done, chunks_scrubbed):
    """
    Test that one chunk is scrubbed by default, and every chunk with --until-done, each starting after the last.
    """
    chunks = [[types.SimpleNamespace(id=1), types.SimpleNamespace(id=2)], [types.SimpleNamespace(id=3)], []]
    after_ids = []
    scrubbed_chunks = []

    def get_files_to_process(args, after_id=None):
        after_ids.append(after_id)
        return chunks[len(scrubbed_chunks)]

    def process_files(args, files, analyzer):
        scrubbed_chunks.append(files)
        return []

    monkeypatch.setattr(pii_scrubber, 'get_files_to_process', get_files_to_process)
    monkeypatch.setattr(pii_scrubber, 'process_files', process_files)
    args = argparse.Namespace(output_to='output', threshold=0.0, until_done=until_done)

    pii_scrubber.scrub_chunks(args, analyzer=None)

    assert scrubbed_chunks == chunks[:chunks_scrubbed]
    assert after_ids == [None, 2, 3][:chunks_scrubbed + until_done]


def test_long_text_is_analyzed_in_pieces():
    """
    Test that no piece given to the analyzer is longer than max_length, and that results land in the right place.
    """
    line = 'Patient seen by Dr. Secret today.\n'
    analyzer = FakeAnalyzer()

    scrubbed, results = pii_scrubber.scrub_pii(line * 100, analyzer, threshold=0.0, max_length=200)

    assert len(analyzer.lengths) > 1
    assert max(analyzer.lengths) <= 200
    assert sum(analyzer.lengths) == len(line) * 100
    assert len(results) == 100
    assert scrubbed == 'Patient seen by Dr. <PERSON> today.\n' * 100


def test_text_over_the_spacy_limit_is_analyzed_in_pieces():
    """
    Test that by default, a text longer than spaCy's 1,000,000 character limit is analyzed in smaller pieces.
    """
    analyzer = FakeAnalyzer()

    pii_scrubber.scrub_pii('Some words on a line.\n' * 55_000, analyzer, threshold=0.0)

    assert max(analyzer.lengths) <= pii_scrubber.MAX_ANALYZE_LENGTH < 1_000_000


def test_missing_status_columns_are_reported(monkeypatch, tmp_path):
    """
    Test that a database created before failures were recorded is reported with the SQL that fixes it.
    """
    db_url = f'sqlite:///{tmp_path}/old.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    with create_engine(db_url).begin() as connection:
        connection.execute(text('CREATE TABLE transcription_output (id INTEGER PRIMARY KEY, '
                                'pii_scrubber_output_file VARCHAR(512))'))
    args = argparse.Namespace(debug_sql=False)

    with pytest.raises(RuntimeError, match=re.escape(pii_scrubber.ADD_STATUS_COLUMNS_SQL)):
        pii_scrubber.check_status_columns(args)


def test_status_columns_check_passes_on_a_current_database(tmp_path, engine):
    """
    Test that a database created with the current schema passes the check.
    """
    pii_scrubber.check_status_columns(argparse.Namespace(debug_sql=False))
