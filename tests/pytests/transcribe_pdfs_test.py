import argparse
import logging

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)
from vmrt_tesseract_utilities.tesseract_operations import NoTextFoundError

"""
Tests for the transcribe_pdfs script.
"""


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    A SQLite database with one good, one blank and one broken PDF waiting to be transcribed.
    """
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


def _run(tmp_path, retry_failed=False):
    args = argparse.Namespace(output_to=str(tmp_path), document_type='document', chunk_size=10,
                              offset=0, retry_failed=retry_failed, debug_sql=False)
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
