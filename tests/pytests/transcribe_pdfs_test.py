import argparse
import logging
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)
from vmrt_tesseract_utilities.tesseract_operations import (NoTextFoundError,
                                                           OcrSetupError)

"""
Tests for the transcribe_pdfs script.
"""


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    A SQLite database with one good, one blank and one broken PDF waiting to be transcribed.

    The OCR setup check is skipped, since poppler and Tesseract aren't installed in CI.
    """
    monkeypatch.setattr(transcribe_pdfs, 'check_ocr_setup', lambda: None)
    db_url = f'sqlite:///{tmp_path}/test.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for name in ('good.pdf', 'blank.pdf', 'broken.pdf'):
            session.add(TranscriptionInput(document_type='document', input_file=name))
        session.commit()
    return engine


def _fake_process_row(self, input_path):
    if input_path == 'blank.pdf':
        raise NoTextFoundError('No recognizable text found in blank.pdf.')
    if input_path == 'broken.pdf':
        raise ValueError('Unable to get page count.')
    return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]


def _run(tmp_path, retry_failed=False, shard=None):
    args = argparse.Namespace(output_to=str(tmp_path), document_type='document', chunk_size=10,
                              offset=0, retry_failed=retry_failed, shard=shard, debug_sql=False)
    transcribe_pdfs.run_tesseract(args)


def _statuses(engine) -> dict:
    with Session(engine) as session:
        return {i.input_file: (i.ocr_status, i.ocr_error) for i in session.query(TranscriptionInput)}


def test_run_tesseract_skips_failures_and_summarizes(monkeypatch, tmp_path, caplog, engine):
    """
    Test that one failing file doesn't stop the run, and the failures are listed at the end.

    Asserts:
        The good file gets an output row and the failed files don't.
        Each input's status and error are saved.
        The summary counts the successes and names each failed file with its error.
    """
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _fake_process_row)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        _run(tmp_path)

    with Session(engine) as session:
        outputs = session.query(TranscriptionOutput).all()
        assert [o.transcription_input.input_file for o in outputs] == ['good.pdf']
    assert _statuses(engine) == {
        'good.pdf': ('success', None),
        'blank.pdf': ('failed', 'NoTextFoundError: No recognizable text found in blank.pdf.'),
        'broken.pdf': ('failed', 'ValueError: Unable to get page count.'),
    }
    assert 'Transcribed 1 of 3 files.' in caplog.text
    assert '2 file(s) failed:' in caplog.text
    assert 'blank.pdf: NoTextFoundError: No recognizable text found in blank.pdf.' in caplog.text
    assert 'broken.pdf: ValueError: Unable to get page count.' in caplog.text


def test_write_error_is_recorded_as_a_failure(monkeypatch, tmp_path, caplog, engine):
    """
    Test that a text file that can't be written is recorded as a failure, and the run carries on.

    Asserts:
        No file gets an output row, each is saved as failed with the write error, and the summary still runs.
    """
    def unwritable(output_path, data):
        raise PermissionError(f'Permission denied: {output_path}')

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row',
                        lambda self, path: [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}])
    monkeypatch.setattr(transcribe_pdfs, '_write_file', unwritable)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        _run(tmp_path)

    with Session(engine) as session:
        assert session.query(TranscriptionOutput).count() == 0
    statuses = _statuses(engine)
    assert {status for status, _ in statuses.values()} == {'failed'}
    assert all(error.startswith('PermissionError: Permission denied') for _, error in statuses.values())
    assert 'Transcribed 0 of 3 files.' in caplog.text


def test_failed_files_are_only_retried_when_asked(monkeypatch, tmp_path, engine):
    """
    Test that files that failed are skipped on the next run, and retried with `--retry-failed`.

    Asserts:
        A plain second run processes nothing.
        A retry run processes only the failed files, and a success clears the saved error.
    """
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _fake_process_row)
    _run(tmp_path)

    seen = []

    def recording_process_row(self, input_path):
        seen.append(input_path)
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', recording_process_row)
    _run(tmp_path)
    assert seen == []

    _run(tmp_path, retry_failed=True)
    assert sorted(seen) == ['blank.pdf', 'broken.pdf']
    assert set(_statuses(engine).values()) == {('success', None)}


def test_setup_error_stops_run_before_any_file(monkeypatch, tmp_path, engine):
    """
    Test that a setup problem stops the run instead of failing every file one by one.

    Asserts:
        OcrSetupError is raised, no file is processed, and no file is marked as failed.
    """
    def broken_setup():
        raise OcrSetupError('Poppler is not installed or not on PATH (missing pdfinfo).')

    def unexpected_process_row(self, input_path):
        raise AssertionError('No file should be processed when setup is broken.')

    monkeypatch.setattr(transcribe_pdfs, 'check_ocr_setup', broken_setup)
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', unexpected_process_row)
    with pytest.raises(OcrSetupError, match='Poppler'):
        _run(tmp_path)
    assert set(_statuses(engine).values()) == {(None, None)}


def _recorder(seen):
    def recording_process_row(self, input_path):
        seen.append(input_path)
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]
    return recording_process_row


def test_shards_split_the_inputs_without_overlap(monkeypatch, tmp_path, engine):
    """
    Test that runs with --shard 0/2 and 1/2 between them transcribe every input exactly once.

    The fixture's inputs have IDs 1 (good.pdf), 2 (blank.pdf) and 3 (broken.pdf). Shard 1/2 runs first, so shard
    0/2 runs after those files are done: with --offset that would move its window, with --shard it must not.

    Asserts:
        Shard 1/2 takes IDs 1 and 3, shard 0/2 takes ID 2, a second round takes nothing, and every input succeeded.
    """
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    _run(tmp_path, shard=(1, 2))
    assert sorted(seen) == ['broken.pdf', 'good.pdf']
    seen.clear()
    _run(tmp_path, shard=(0, 2))
    assert seen == ['blank.pdf']
    seen.clear()
    _run(tmp_path, shard=(0, 2))
    _run(tmp_path, shard=(1, 2))
    assert seen == []
    assert set(_statuses(engine).values()) == {('success', None)}


def test_no_shard_takes_every_input(monkeypatch, tmp_path, engine):
    """
    Test that a run without --shard still transcribes every input.

    Asserts:
        All three inputs are processed.
    """
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    _run(tmp_path)
    assert sorted(seen) == ['blank.pdf', 'broken.pdf', 'good.pdf']


@pytest.mark.parametrize('value, expected', [('0/1', (0, 1)), ('3/4', (3, 4)), (' 1 / 2 ', (1, 2))])
def test_parse_shard_accepts_valid_values(value, expected):
    """
    Test that "i/N" values with 0 <= i < N are parsed.

    Asserts:
        The value is returned as (i, N).
    """
    assert transcribe_pdfs.parse_shard(value) == expected


@pytest.mark.parametrize('value, message', [
    ('2', 'expected i/N'),
    ('a/4', 'expected i/N'),
    ('1/2/3', 'expected i/N'),
    ('0/0', 'at least 1'),
    ('4/4', 'from 0 to 3'),
    ('-1/4', 'from 0 to 3'),
])
def test_parse_shard_rejects_invalid_values(value, message):
    """
    Test that a --shard value that isn't "i/N" with 0 <= i < N is rejected with a clear message.

    Asserts:
        argparse.ArgumentTypeError is raised, naming the problem.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        transcribe_pdfs.parse_shard(value)


def test_shard_flag_is_parsed(monkeypatch):
    """
    Test that --shard reaches the parsed args as (i, N), and defaults to None.

    Asserts:
        `--shard 1/3` gives (1, 3); no flag gives None; a bad value exits with an argparse error.
    """
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output', '--shard', '1/3'])
    assert transcribe_pdfs.parse_args().shard == (1, 3)
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output'])
    assert transcribe_pdfs.parse_args().shard is None
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output', '--shard', '3/3'])
    with pytest.raises(SystemExit):
        transcribe_pdfs.parse_args()
