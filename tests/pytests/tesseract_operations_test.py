import os
import re
import shutil
from pathlib import Path

import pytest

"""
Tests for the Tesseract operations, run on a small generated multi-page PDF.

Tesseract and poppler are only installed in the Docker image, so these are skipped elsewhere (including CI).
"""

tesserocr = pytest.importorskip('tesserocr')
pdf2image = pytest.importorskip('pdf2image')
if shutil.which('pdfinfo') is None or not os.path.isdir('/usr/share/tessdata'):
    pytest.skip('poppler or the Tesseract models are not installed', allow_module_level=True)

from tests.synthetic_pdfs import SyntheticPage, write_pdf  # noqa: E402
from vmrt_tesseract_utilities import tesseract_operations  # noqa: E402
from vmrt_tesseract_utilities.tesseract_operations import (  # noqa: E402
    TesseractOperationBase, TesseractOperationBlock, TesseractOperationDoc,
    TesseractOperationPage)

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
    path = str(tmp_path_factory.mktemp('ocr') / 'synthetic.pdf')
    return write_pdf(path, [SyntheticPage(lines) for lines in PAGE_LINES])


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


def test_large_page_is_rendered_within_pixel_limit(pdf_path, monkeypatch):
    """
    Test that a page larger than MAX_PAGE_PIXELS at 200 dpi is rendered smaller, and its text is still read.

    The limit is lowered so the synthetic letter-size pages count as too large, which keeps the test fast.
    """
    monkeypatch.setattr(tesseract_operations, 'MAX_PAGE_PIXELS', 1_000_000)
    sizes = [image.size for _, image in TesseractOperationBase._iter_pages(pdf_path)]
    assert all(width * height <= 1_000_000 for width, height in sizes)
    # About 100 dpi rather than 200: 1700 x 2200 pixels.
    assert all(800 < width < 900 for width, _ in sizes)
    output = TesseractOperationDoc().process_row(pdf_path)
    for lines in PAGE_LINES:
        for line in lines:
            assert line in output[0]['content']


def test_long_page_is_rendered_within_side_limit(pdf_path, monkeypatch):
    """
    Test that a page with a side longer than MAX_PAGE_SIDE at 200 dpi is rendered smaller, and its text is still
    read.

    The limit is lowered so the synthetic letter-size pages count as too long, which keeps the test fast.
    """
    monkeypatch.setattr(tesseract_operations, 'MAX_PAGE_SIDE', 1500)
    sizes = [image.size for _, image in TesseractOperationBase._iter_pages(pdf_path)]
    assert all(max(width, height) <= 1500 for width, height in sizes)
    output = TesseractOperationDoc().process_row(pdf_path)
    for lines in PAGE_LINES:
        for line in lines:
            assert line in output[0]['content']


def test_max_page_pixels_matches_pillow():
    """
    Test that MAX_PAGE_PIXELS, set as a number because CI replaces Pillow, is still the size Pillow refuses above, so
    a Pillow upgrade that changes its limit fails here rather than on real records.
    """
    from PIL import Image
    assert tesseract_operations.MAX_PAGE_PIXELS == 2 * Image.MAX_IMAGE_PIXELS


def _with_crop_box(source: str, path, crop_box: str) -> str:
    """
    Copies a one-page PDF written by Pillow, adding a CropBox to its page. Poppler rebuilds the cross-reference
    table the insertion invalidates.
    """
    data = Path(source).read_bytes()
    media_box = re.search(rb'/MediaBox \[[^\]]*\]', data).group(0)
    path.write_bytes(data.replace(media_box, media_box + b' /CropBox [ ' + crop_box.encode() + b' ]', 1))
    return str(path)


@pytest.mark.parametrize('crop_box', ['0 0 306 396', '700 0 800 100'])
def test_page_size_is_the_rendered_media_box(tmp_path, crop_box):
    """
    Test that page sizes come from the MediaBox, which pdf2image renders, not the CropBox pdfinfo reports as the
    page size: a smaller one, or one outside the page that poppler clips to zero width.
    """
    source = write_pdf(str(tmp_path / 'letter.pdf'), [SyntheticPage(['Vaccines current'])])
    path = _with_crop_box(source, tmp_path / 'cropped.pdf', crop_box)
    assert tesseract_operations.page_sizes(path, 1) == {1: (8.5, 11.0)}
    output = TesseractOperationDoc().process_row(path)
    assert 'Vaccines current' in output[0]['content']


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / 'broken.pdf'
    path.write_text('not a pdf')
    for op in (TesseractOperationDoc(), TesseractOperationPage(), TesseractOperationBlock()):
        with pytest.raises(Exception):
            op.process_row(str(path))


@pytest.mark.parametrize('op', [TesseractOperationDoc(), TesseractOperationPage(), TesseractOperationBlock()])
def test_failure_part_way_through_raises(pdf_path, monkeypatch, op):
    """
    Test that a page failing to render raises, rather than returning the pages before it, which would be saved
    as if they were the whole file.
    """
    convert_from_path = pdf2image.convert_from_path

    def fail_on_second_page(path, **kwargs):
        if kwargs.get('first_page') == 2:
            raise RuntimeError('simulated render failure')
        return convert_from_path(path, **kwargs)

    monkeypatch.setattr(pdf2image, 'convert_from_path', fail_on_second_page)
    with pytest.raises(RuntimeError, match='simulated render failure'):
        op.process_row(pdf_path)
