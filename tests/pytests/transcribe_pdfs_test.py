"""Regression tests for one database row per page/block OCR output (issue #49)."""

import argparse

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (
    Base, TranscriptionInput, TranscriptionOutput,
)


def _make_two_page_pdf(path):
    """Use synthetic pages, never an actual veterinary record."""
    pages = []
    for page_number in (1, 2):
        image = Image.new('RGB', (400, 200), 'white')
        ImageDraw.Draw(image).text((15, 30), f'Synthetic page {page_number}', fill='black')
        pages.append(image)
    pages[0].save(path, 'PDF', save_all=True, append_images=pages[1:])


@pytest.mark.parametrize(
    'mode,results',
    [
        ('page', [
            {'content': 'Page one', 'confidence': 91.0, 'page': 1, 'block': 0},
            {'content': 'Page two', 'confidence': 92.0, 'page': 2, 'block': 0},
        ]),
        ('block', [
            {'content': 'P1 block 1', 'confidence': 81.0, 'page': 1, 'block': 1},
            {'content': 'P1 block 2', 'confidence': 82.0, 'page': 1, 'block': 2},
            {'content': 'P1 block 3', 'confidence': 83.0, 'page': 1, 'block': 3},
            {'content': 'P2 block 1', 'confidence': 84.0, 'page': 2, 'block': 1},
            {'content': 'P2 block 2', 'confidence': 85.0, 'page': 2, 'block': 2},
            {'content': 'P2 block 3', 'confidence': 86.0, 'page': 2, 'block': 3},
        ]),
    ],
)
@pytest.mark.parametrize('has_existing_empty_row', [False, True])
def test_every_ocr_result_has_its_own_database_row(
    monkeypatch, tmp_path, mode, results, has_existing_empty_row
):
    pdf_path = tmp_path / 'synthetic.pdf'
    _make_two_page_pdf(pdf_path)

    db_url = f'sqlite:///{tmp_path / "test.db"}'
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    old_row_id = None
    with Session(engine) as session:
        input_row = TranscriptionInput(document_type=mode, input_file=str(pdf_path))
        session.add(input_row)
        if has_existing_empty_row:
            old_row = TranscriptionOutput(transcription_input=input_row)
            session.add(old_row)
        session.commit()
        if has_existing_empty_row:
            old_row_id = old_row.id

    def fake_process_row(self, input_path):
        assert input_path == str(pdf_path)
        return results

    operation = (
        transcribe_pdfs.TesseractOperationPage if mode == 'page'
        else transcribe_pdfs.TesseractOperationBlock
    )
    monkeypatch.setattr(operation, 'process_row', fake_process_row)
    monkeypatch.setattr(
        transcribe_pdfs, 'get_database_session',
        lambda **kwargs: __import__('sqlalchemy.orm', fromlist=['sessionmaker']).sessionmaker(engine),
    )

    args = argparse.Namespace(
        output_to=str(tmp_path), document_type=mode,
        chunk_size=10, offset=0, debug_sql=False,
    )
    transcribe_pdfs.run_tesseract(args)

    with Session(engine) as session:
        stored_rows = session.scalars(
            select(TranscriptionOutput).order_by(TranscriptionOutput.id)
        ).all()
        assert len(stored_rows) == len(results)
        assert len({row.ocr_output_file for row in stored_rows}) == len(results)
        assert {row.ocr_confidence for row in stored_rows} == {
            result['confidence'] for result in results
        }
        assert all(row.transcription_input.input_file == str(pdf_path) for row in stored_rows)
        if has_existing_empty_row:
            assert old_row_id in {row.id for row in stored_rows}

        by_file = {row.ocr_output_file: row for row in stored_rows}
        for result in results:
            file_path = (
                tmp_path / 'unstructured_text' / mode /
                f"synthetic-{result['page']}-{result['block']}.txt"
            )
            assert str(file_path) in by_file
            assert by_file[str(file_path)].ocr_confidence == result['confidence']
            assert file_path.read_text() == result['content']

    engine.dispose()
