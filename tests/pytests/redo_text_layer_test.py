"""
Tests for redo_text_layer.py, which sends PDFs transcribed before issue #82 back through the pipeline when some of
their pages would now be read from their text layer.

The decision per PDF is faked, so these run without poppler; the last test makes it on real generated PDFs, inside
the Docker image only. All of the text is made up.
"""

import argparse
import datetime
import logging
import shutil
from unittest import mock

import PIL
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scripts import redo_text_layer
from vmrt_tesseract_utilities.database import (Base,
                                               StringReplacementCompletion,
                                               StringReplacementPass,
                                               TranscriptionInput,
                                               TranscriptionMetadata,
                                               TranscriptionOutput)

DECISIONS = {
    'digital.pdf': (redo_text_layer.REDO, '2 of 2 pages from the text layer'),
    'scan.pdf': (redo_text_layer.KEEP, '0 of 3 pages from the text layer'),
    'odd.pdf': (redo_text_layer.UNREADABLE, 'pdfinfo exited with code 1 on odd.pdf'),
}


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    A SQLite database of a pipeline that ran before #82:

    - digital.pdf and scan.pdf went through every stage: two output rows each, with their files, a string
      replacement record each, and a visit date found;
    - odd.pdf was transcribed but its text layer can't be read;
    - failed.pdf failed OCR, and new.pdf was transcribed after #82 (its rows have a text_source).
    """
    monkeypatch.setattr(redo_text_layer, 'decide', lambda path: DECISIONS[path])
    db_url = f'sqlite:///{tmp_path}/test.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        replacement_pass = StringReplacementPass(pass_hash='abc', replacement_string='<NAME>', source_file='dogs.tsv',
                                                 key_column='name', target_count=1)
        session.add(replacement_pass)
        for name, status in (('digital.pdf', 'success'), ('scan.pdf', 'success'), ('odd.pdf', 'success'),
                             ('failed.pdf', 'failed'), ('new.pdf', 'success')):
            db_input = TranscriptionInput(document_type='page', input_file=name, ocr_status=status)
            session.add(db_input)
            db_input.derived_metadata.append(TranscriptionMetadata(
                subject_id='094-000001', visit_date=datetime.datetime(2023, 3, 14),
                extracted_date=datetime.datetime(2023, 3, 14)))
            db_input.derived_metadata.append(TranscriptionMetadata(
                subject_id='094-000001', visit_date=datetime.datetime(2024, 3, 14),
                extracted_date=datetime.datetime(2024, 3, 15)))
            if status == 'failed':
                continue
            for page in (1, 2):
                files = []
                for stage in ('ocr', 'replaced', 'scrubbed', 'confidence'):
                    path = tmp_path / f'{name}-{page}-{stage}.txt'
                    path.write_text('old text')
                    files.append(str(path))
                output = TranscriptionOutput(ocr_output_file=files[0], ocr_confidence=88,
                                             list_replacement_output_file=files[1], pii_scrubber_output_file=files[2],
                                             pii_scrubber_confidence_file=files[3], pii_scrubber_status='success',
                                             text_source='ocr' if name == 'new.pdf' else None)
                db_input.assets.append(output)
                session.flush()
                session.add(StringReplacementCompletion(output_id=output.id, pass_id=replacement_pass.id))
        session.commit()
    return engine


def _run(dry_run=False, chunk_size=500):
    args = argparse.Namespace(document_type='page', dry_run=dry_run, chunk_size=chunk_size, workers=1,
                              debug_sql=False)
    return redo_text_layer.redo(args)


def _state(engine) -> dict:
    with Session(engine) as session:
        return {i.input_file: (i.ocr_status, len(i.assets), sorted({o.text_source for o in i.assets}, key=str),
                               sorted((m.visit_date is not None) for m in i.derived_metadata))
                for i in session.query(TranscriptionInput)}


def test_redo_resets_born_digital_pdfs_and_marks_the_rest(engine, tmp_path, caplog):
    """
    Test that a PDF with text-layer pages is sent back to the start, and the others are marked as checked.

    Asserts:
        digital.pdf has no outputs, string replacement records or output files left, one empty metadata row, and no
        ocr_status, so transcribe_pdfs.py picks it up again.
        scan.pdf and odd.pdf keep their outputs and files, marked text_source 'ocr'.
        failed.pdf and new.pdf aren't touched.
        The summary counts each.
    """
    with caplog.at_level(logging.INFO, logger='import_utility'):
        counts = _run()
    assert counts == {redo_text_layer.REDO: 1, redo_text_layer.KEEP: 1, redo_text_layer.UNREADABLE: 1}
    assert _state(engine) == {
        'digital.pdf': (None, 0, [], [False]),
        'scan.pdf': ('success', 2, ['ocr'], [True, True]),
        'odd.pdf': ('success', 2, ['ocr'], [True, True]),
        'failed.pdf': ('failed', 0, [], [True, True]),
        'new.pdf': ('success', 2, ['ocr'], [True, True]),
    }
    with Session(engine) as session:
        assert session.query(StringReplacementCompletion).count() == 6
        [metadata] = session.query(TranscriptionInput).filter_by(input_file='digital.pdf').one().derived_metadata
        assert (metadata.subject_id, metadata.visit_date, metadata.extracted_date) == ('094-000001', None, None)
    assert not list(tmp_path.glob('digital.pdf-*'))
    assert len(list(tmp_path.glob('scan.pdf-*'))) == 8
    assert 'Redoing digital.pdf (2 of 2 pages from the text layer).' in caplog.text
    assert 'odd.pdf: kept as it is, its text layer could not be read' in caplog.text
    assert 'Redid 1 PDFs; 1 kept their OCR' in caplog.text


def test_dry_run_changes_nothing(engine, tmp_path, caplog):
    """
    Test that --dry-run only counts and logs.

    Asserts:
        The counts are those of a real run, the database and files are as they were, and the log says what would be
        redone.
    """
    before = _state(engine)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        counts = _run(dry_run=True)
    assert counts[redo_text_layer.REDO] == 1
    assert _state(engine) == before
    assert len(list(tmp_path.glob('*.pdf-*'))) == 32
    assert 'Would redo digital.pdf (2 of 2 pages from the text layer).' in caplog.text


def test_rerun_checks_nothing_twice(engine):
    """
    Test that the script can be run again (after being stopped, say) without redoing or rechecking anything.

    Asserts:
        A second run finds no PDFs left to check.
    """
    _run(chunk_size=1)
    assert _run() == {redo_text_layer.REDO: 0, redo_text_layer.KEEP: 0, redo_text_layer.UNREADABLE: 0}


def test_missing_output_file_is_fine(engine, tmp_path):
    """
    Test that an output file that's already gone doesn't stop a PDF from being redone.
    """
    (tmp_path / 'digital.pdf-1-ocr.txt').unlink()
    _run()
    assert _state(engine)['digital.pdf'] == (None, 0, [], [False])


@pytest.mark.skipif(shutil.which('pdftotext') is None or isinstance(PIL, mock.MagicMock),
                    reason='poppler or Pillow is not installed')
def test_decide_on_real_pdfs(tmp_path):
    """
    Test the decision on generated PDFs: redo one with a born-digital page, keep a scan, and keep a damaged file.
    """
    from tests.synthetic_pdfs import DigitalPage, SyntheticPage, write_pdf
    lines = ['Annual wellness exam for a nine year old spayed female.',
             'Owner reports good appetite, normal energy and no vomiting.',
             'Exam: Weight 31.2 kg. Temperature 101.4 F. Heart rate 96 bpm.']
    scanned = ['Recheck exam, ears clean.', 'Weight 30.8 kg.']
    mixed = write_pdf(str(tmp_path / 'mixed.pdf'), [DigitalPage(lines), SyntheticPage(scanned)])
    scan = write_pdf(str(tmp_path / 'scan.pdf'), [SyntheticPage(scanned)])
    broken = tmp_path / 'broken.pdf'
    broken.write_bytes(b'%PDF-1.4\nnot a PDF\n')
    assert redo_text_layer.decide(mixed) == (redo_text_layer.REDO, '1 of 2 pages from the text layer')
    assert redo_text_layer.decide(scan) == (redo_text_layer.KEEP, '0 of 1 pages from the text layer')
    assert redo_text_layer.decide(str(broken))[0] == redo_text_layer.UNREADABLE
