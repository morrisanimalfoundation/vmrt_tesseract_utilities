import os
import shutil

import pytest

"""
Tests for the Tesseract operations, run on a small generated multi-page PDF.

Tesseract and poppler are only installed in the Docker image, so these are skipped elsewhere (including CI).
"""

tesserocr = pytest.importorskip('tesserocr')
pytest.importorskip('pdf2image')
if shutil.which('pdfinfo') is None or not os.path.isdir('/usr/share/tessdata'):
    pytest.skip('poppler or the Tesseract models are not installed', allow_module_level=True)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from vmrt_tesseract_utilities.tesseract_operations import (  # noqa: E402
    TesseractOperationBlock, TesseractOperationDoc, TesseractOperationPage)

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PAGE_LINES = [
    ['Vaccines current', 'Weight 32 kg'],
    ['Heart rate normal', 'Teeth cleaned'],
    ['Recheck in six months'],
]


@pytest.fixture(scope='module')
def pdf_path(tmp_path_factory) -> str:
    """
    Writes a three page PDF of synthetic text, one line of text per `PAGE_LINES` entry.
    """
    font = ImageFont.truetype(FONT_PATH, 48)
    pages = []
    for lines in PAGE_LINES:
        image = Image.new('RGB', (1700, 2200), 'white')
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            draw.text((150, 200 + i * 150), line, fill='black', font=font)
        pages.append(image)
    path = str(tmp_path_factory.mktemp('ocr') / 'synthetic.pdf')
    pages[0].save(path, 'PDF', resolution=200, save_all=True, append_images=pages[1:])
    return path


def test_document_mode(pdf_path):
    output = TesseractOperationDoc().process_row(pdf_path)
    assert len(output) == 1
    assert output[0]['page'] == 1 and output[0]['block'] == 1
    for lines in PAGE_LINES:
        for line in lines:
            assert line in output[0]['content']
    assert output[0]['confidence'] > 50


def test_page_mode(pdf_path):
    output = TesseractOperationPage().process_row(pdf_path)
    assert [o['page'] for o in output] == [1, 2, 3]
    for result, lines in zip(output, PAGE_LINES):
        assert result['block'] == 0
        for line in lines:
            assert line in result['content']
        assert result['confidence'] > 50


def test_block_mode(pdf_path):
    output = TesseractOperationBlock().process_row(pdf_path)
    expected = [(pg + 1, i + 1, line) for pg, lines in enumerate(PAGE_LINES) for i, line in enumerate(lines)]
    assert [(o['page'], o['block'], o['content'].strip()) for o in output] == expected
    for result in output:
        assert result['confidence'] > 50


def test_unreadable_file_returns_nothing(tmp_path):
    path = tmp_path / 'broken.pdf'
    path.write_text('not a pdf')
    for op in (TesseractOperationDoc(), TesseractOperationPage(), TesseractOperationBlock()):
        assert op.process_row(str(path)) == []
