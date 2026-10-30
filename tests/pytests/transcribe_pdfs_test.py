"""Regression tests for one database row per page/block OCR output (issue #49)."""

import argparse

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)


def _make_two_page_pdf(path):
    """Generate a valid two-page PDF using only Python's standard library."""
    def stream(text):
        content = f'BT /F1 20 Tf 25 125 Td ({text}) Tj ET\n'.encode('ascii')
        return b'<< /Length %d >>\nstream\n' % len(content) + content + b'endstream'

    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] '
        b'/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] '
        b'/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        stream('Synthetic page one'),
        stream('Synthetic page two'),
    ]
    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for number, item in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f'{number} 0 obj\n'.encode('ascii') + item + b'\nendobj\n')
    startxref = len(pdf)
    pdf.extend(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
    pdf.extend(
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>'
        f'\nstartxref\n{startxref}\n%%EOF\n'.encode('ascii')
    )
    path.write_bytes(pdf)


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
        lambda **kwargs: sessionmaker(engine),
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
            file_path = tmp_path.joinpath(
                'unstructured_text', mode,
                f"synthetic-{result['page']}-{result['block']}.txt",
            )
            assert str(file_path) in by_file
            assert by_file[str(file_path)].ocr_confidence == result['confidence']
            assert file_path.read_text() == result['content']

    engine.dispose()
