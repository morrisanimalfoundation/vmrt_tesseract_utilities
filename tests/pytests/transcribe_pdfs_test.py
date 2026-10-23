import argparse
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)
from vmrt_tesseract_utilities.tesseract_operations import NoTextFoundError

"""
Tests for the transcribe_pdfs script.
"""


def test_run_tesseract_skips_failures_and_summarizes(monkeypatch, tmp_path, caplog):
    """
    Test that one failing file doesn't stop the run, and the failures are listed at the end.

    Asserts:
        The good file gets an output row and the failed files don't.
        The summary counts the successes and names each failed file with its error.
    """
    db_url = f'sqlite:///{tmp_path}/test.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for name in ('good.pdf', 'blank.pdf', 'broken.pdf'):
            session.add(TranscriptionInput(document_type='document', input_file=name))
        session.commit()

    def fake_process_row(self, input_path):
        if input_path == 'blank.pdf':
            raise NoTextFoundError('No recognizable text found in blank.pdf.')
        if input_path == 'broken.pdf':
            raise ValueError('Unable to get page count.')
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', fake_process_row)
    args = argparse.Namespace(output_to=str(tmp_path), document_type='document',
                              chunk_size=10, offset=0, debug_sql=False)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        transcribe_pdfs.run_tesseract(args)

    with Session(engine) as session:
        outputs = session.query(TranscriptionOutput).all()
        assert [o.transcription_input.input_file for o in outputs] == ['good.pdf']
    assert 'Transcribed 1 of 3 files.' in caplog.text
    assert '2 file(s) failed:' in caplog.text
    assert 'blank.pdf: NoTextFoundError: No recognizable text found in blank.pdf.' in caplog.text
    assert 'broken.pdf: ValueError: Unable to get page count.' in caplog.text
