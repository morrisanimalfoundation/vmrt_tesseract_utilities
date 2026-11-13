"""
Tests for how the `pii_scrubber` script picks files, records failures and splits long texts.

These use a stand-in for the analyzer, so they need Presidio's packages but not the model. The packages are only
installed in the workspace image, so the tests are skipped elsewhere.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

for module in ('presidio_analyzer', 'presidio_anonymizer'):
    pytest.importorskip(module, reason='The PII scrubber packages are only installed in the workspace image.')

from presidio_analyzer import RecognizerResult  # noqa: E402

from scripts.scrubbers import pii_scrubber  # noqa: E402
from vmrt_tesseract_utilities import database  # noqa: E402
from vmrt_tesseract_utilities.cli import EXIT_TOO_MANY_FAILURES  # noqa: E402


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
    _, failures = pii_scrubber.scrub_chunks(args, analyzer)
    return failures


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
    Test that one chunk is scrubbed by default, and every chunk with --until-done, leaving out files that failed.
    """
    chunks = [['first', 'second'], ['third'], []]
    excluded = []
    scrubbed_chunks = []

    def get_files_to_process(args, failed_ids=frozenset()):
        excluded.append(failed_ids)
        return chunks[len(scrubbed_chunks)]

    def process_files(args, files, analyzer, failed_ids):
        scrubbed_chunks.append(files)
        failed_ids.add(len(scrubbed_chunks))
        return []

    monkeypatch.setattr(pii_scrubber, 'get_files_to_process', get_files_to_process)
    monkeypatch.setattr(pii_scrubber, 'process_files', process_files)
    args = argparse.Namespace(output_to='output', threshold=0.0, until_done=until_done)

    count, _ = pii_scrubber.scrub_chunks(args, analyzer=None)

    assert count == sum(len(chunk) for chunk in chunks[:chunks_scrubbed])
    assert scrubbed_chunks == chunks[:chunks_scrubbed]
    assert excluded == [frozenset(), {1}, {1, 2}][:chunks_scrubbed + until_done]


def test_file_ready_part_way_through_a_run_is_scrubbed(tmp_path, engine):
    """
    Test that with --until-done, a file whose OCR finishes during the run is scrubbed even if its ID is lower than
    files already scrubbed.
    """
    with Session(engine) as session:
        late = database.TranscriptionOutput(ocr_output_file=None)
        session.add(database.TranscriptionInput(document_type='document', input_file='late.pdf', assets=[late]))
        session.commit()
        # Give the late file the lowest ID, below the files the run starts with.
        session.execute(text('UPDATE transcription_output SET id = 0 WHERE id = :id'), {'id': late.id})
        session.commit()
    (tmp_path / 'late.txt').write_text('Late.\n')

    class OcrFinishesDuringRun(FakeAnalyzer):
        def analyze(self, text, language):
            if text == 'No names.\n':
                with Session(engine) as session:
                    session.execute(sqlalchemy.text('UPDATE transcription_output SET ocr_output_file = :path WHERE id = 0'),
                                    {'path': str(tmp_path / 'late.txt')})
                    session.commit()
            return super().analyze(text, language)

    _run(tmp_path, OcrFinishesDuringRun(), chunk_size=1)

    assert _statuses(engine)['late.pdf'] == ('success', None, True)


class _Killed(BaseException):
    """
    Stands in for the process being killed part way through a file: like a SIGKILL, it isn't caught by the script's
    `except Exception`, so nothing after it is recorded, but what was already committed stays.
    """


class KilledOn(FakeAnalyzer):
    """
    A FakeAnalyzer that's "killed" on text containing the given word.
    """

    def __init__(self, word):
        super().__init__()
        self.word = word

    def analyze(self, text, language):
        if self.word in text:
            self.lengths.append(len(text))
            raise _Killed()
        return super().analyze(text, language)


def _kill_on_other(tmp_path):
    (tmp_path / 'other.txt').write_text('Killer text.\n')
    with pytest.raises(_Killed):
        _run(tmp_path, KilledOn('Killer'))


def test_file_is_marked_started_before_scrubbing(tmp_path, engine):
    """
    Test that a file is saved as started before it's scrubbed, so a run killed part way through it leaves a record.
    """
    _kill_on_other(tmp_path)

    assert _statuses(engine)['other.pdf'] == ('started', None, False)


def test_file_cut_short_twice_is_marked_failed(tmp_path, engine, caplog):
    """
    Test that a file cut short twice is tried once more as retrying, then marked failed without being read, and
    skipped after that.
    """
    _kill_on_other(tmp_path)
    with pytest.raises(_Killed):
        _run(tmp_path, KilledOn('Killer'))
    assert _statuses(engine)['other.pdf'] == ('retrying', None, False)

    analyzer = KilledOn('Killer')
    failures = _run(tmp_path, analyzer)
    assert analyzer.lengths == []
    assert [path for path, _ in failures] == [str(tmp_path / 'other.txt')]
    assert _statuses(engine)['other.pdf'] == ('failed', pii_scrubber.INTERRUPTED_TWICE_ERROR, False)

    assert _run(tmp_path, analyzer) == []
    assert analyzer.lengths == []


def test_file_cut_short_twice_leaves_no_output(tmp_path, engine):
    """
    Test that a file marked failed after being cut short twice doesn't leave behind outputs the killed runs wrote.
    """
    _kill_on_other(tmp_path)
    with pytest.raises(_Killed):
        _run(tmp_path, KilledOn('Killer'))
    args = argparse.Namespace(output_to=str(tmp_path / 'output'), document_type='document')
    outputs = [Path(path) for path in pii_scrubber.output_paths(args, str(tmp_path / 'other.txt'))]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Partly written.\n')

    _run(tmp_path, KilledOn('Killer'))

    assert _statuses(engine)['other.pdf'] == ('failed', pii_scrubber.INTERRUPTED_TWICE_ERROR, False)
    assert not any(path.exists() for path in outputs)


def test_file_cut_short_once_succeeds_on_the_next_run(tmp_path, engine, caplog):
    """
    Test that a file cut short once is scrubbed on the next run, with a warning that it's being tried again.
    """
    _kill_on_other(tmp_path)

    with caplog.at_level(logging.INFO):
        _run(tmp_path, FakeAnalyzer())

    assert _statuses(engine)['other.pdf'] == ('success', None, True)
    assert 'an earlier run was cut short while scrubbing it' in caplog.text


def test_retry_failed_scrubs_a_retrying_file(tmp_path, engine):
    """
    Test that --retry-failed scrubs a file left retrying instead of marking it failed unread.
    """
    _kill_on_other(tmp_path)
    with pytest.raises(_Killed):
        _run(tmp_path, KilledOn('Killer'))

    _run(tmp_path, FakeAnalyzer(), retry_failed=True)

    assert _statuses(engine)['other.pdf'] == ('success', None, True)


def test_retrying_a_failed_file_clears_its_old_error(tmp_path, engine):
    """
    Test that a failed file being scrubbed again doesn't keep the error from its last attempt.
    """
    _run(tmp_path, FakeAnalyzer())
    (tmp_path / 'bad.txt').write_text('Killer text.\n')

    with pytest.raises(_Killed):
        _run(tmp_path, KilledOn('Killer'), retry_failed=True)

    assert _statuses(engine)['bad.pdf'] == ('started', None, False)


def test_long_error_is_cut_short_when_saved(tmp_path, engine):
    """
    Test that an error too long for the database column is saved cut short, and listed in full.
    """
    class LongError(FakeAnalyzer):
        def analyze(self, text, language):
            raise ValueError('x' * 100_000)

    failures = _run(tmp_path, LongError())

    assert len(failures[0][1]) > 100_000
    saved = _statuses(engine)['good.pdf'][1]
    assert saved.startswith('ValueError: xxx') and saved.endswith('... (cut short)')
    assert len(saved) <= pii_scrubber.MAX_ERROR_LENGTH + len('... (cut short)')


def test_failure_that_cant_be_saved_doesnt_stop_the_run(monkeypatch, tmp_path, engine, caplog):
    """
    Test that if saving a failure fails, that's logged and the other files are still scrubbed.
    """
    real_commit = Session.commit

    def commit(self):
        if any(getattr(row, 'pii_scrubber_status', None) == 'failed' for row in self.dirty):
            raise OperationalError('UPDATE', {}, Exception('Lost connection'))
        real_commit(self)

    monkeypatch.setattr(Session, 'commit', commit)

    failures = _run(tmp_path, FakeAnalyzer())

    assert len(failures) == 1
    assert 'Could not save the failure of' in caplog.text
    statuses = _statuses(engine)
    assert statuses['bad.pdf'] == ('started', None, False)
    assert statuses['other.pdf'] == ('success', None, True)


def test_failed_file_leaves_no_output(monkeypatch, tmp_path, engine):
    """
    Test that a file that fails after its scrubbed text was written doesn't leave that text behind.
    """
    def write_confidence_record(filename, filtered_results, original_text):
        raise OSError('No space left on device')

    monkeypatch.setattr(pii_scrubber, 'write_confidence_record', write_confidence_record)

    _run(tmp_path, FakeAnalyzer())

    assert _statuses(engine)['good.pdf'] == ('failed', 'OSError: No space left on device', False)
    assert not list((tmp_path / 'output').rglob('*.txt'))


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


def test_missing_table_is_reported(monkeypatch, tmp_path):
    """
    Test that a database without the pipeline's tables is reported with what to run, not a traceback.
    """
    monkeypatch.setenv('SQL_CONNECTION_STRING', f'sqlite:///{tmp_path}/empty.db')

    with pytest.raises(RuntimeError, match='database_setup.py install'):
        pii_scrubber.check_status_columns(argparse.Namespace(debug_sql=False))


def test_status_columns_check_passes_on_a_current_database(tmp_path, engine):
    """
    Test that a database created with the current schema passes the check.
    """
    pii_scrubber.check_status_columns(argparse.Namespace(debug_sql=False))


def _main(monkeypatch, tmp_path, *flags):
    """
    Runs the script's main() with the stand-in analyzer, as `pii_scrubber.py <output> --until-done <flags>`.
    """
    monkeypatch.setattr(pii_scrubber, 'create_nlp_engine', lambda config: FakeAnalyzer())
    monkeypatch.setattr(sys, 'argv', ['pii_scrubber.py', str(tmp_path / 'output'), '--until-done', *flags])
    return pii_scrubber.main()


def test_main_exits_with_3_when_every_file_fails(monkeypatch, tmp_path, engine, caplog):
    """
    Test that a run where every file failed exits with 3, so it doesn't look successful.
    """
    for name in ('good', 'other'):
        (tmp_path / f'{name}.txt').write_text('BROKEN\n')

    with caplog.at_level(logging.INFO):
        assert _main(monkeypatch, tmp_path) == EXIT_TOO_MANY_FAILURES

    assert 'Scrubbed 0 of 3 files.' in caplog.text
    assert 'All 3 files failed' in caplog.text


def test_main_exits_with_0_when_some_files_fail(monkeypatch, tmp_path, engine):
    """
    Test that a run where only some files failed still exits with 0 by default.
    """
    assert _main(monkeypatch, tmp_path) == 0
    assert _statuses(engine)['bad.pdf'][0] == 'failed'


def test_main_exits_with_3_when_more_than_max_failure_rate_fail(monkeypatch, tmp_path, engine):
    """
    Test that with --max-failure-rate, a run where a larger share of files failed exits with 3.
    """
    assert _main(monkeypatch, tmp_path, '--max-failure-rate', '0.2') == EXIT_TOO_MANY_FAILURES


def test_main_exits_with_0_when_every_file_is_scrubbed(monkeypatch, tmp_path, engine):
    """
    Test that a run where every file was scrubbed exits with 0.
    """
    (tmp_path / 'bad.txt').write_text('Fixed.\n')

    assert _main(monkeypatch, tmp_path) == 0
    assert {status for status, _, _ in _statuses(engine).values()} == {'success'}


def test_main_exits_with_0_when_there_are_no_files(monkeypatch, tmp_path, engine, caplog):
    """
    Test that a run with no files to scrub exits with 0.
    """
    with caplog.at_level(logging.INFO):
        assert _main(monkeypatch, tmp_path, '--document-type', 'block') == 0

    assert 'Found 0 files to scrub.' in caplog.text


def test_main_exits_with_1_when_it_cant_run(monkeypatch, tmp_path, engine):
    """
    Test that a run that stops before scrubbing anything exits with 1.
    """
    def check_status_columns(args):
        raise RuntimeError('The database has no transcription_output table.')

    monkeypatch.setattr(pii_scrubber, 'check_status_columns', check_status_columns)

    assert _main(monkeypatch, tmp_path) == 1
    assert _statuses(engine)['good.pdf'] == (None, None, False)
