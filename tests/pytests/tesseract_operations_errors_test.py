import logging
import subprocess
from unittest import mock

import pytest

from vmrt_tesseract_utilities import tesseract_operations
from vmrt_tesseract_utilities.tesseract_operations import (
    SOURCE_MIXED, SOURCE_OCR, SOURCE_TEXT_LAYER, NoTextFoundError,
    OcrSetupError, PageTooLargeError, TesseractOperationDoc,
    TesseractOperationPage, check_ocr_setup)
from vmrt_tesseract_utilities.text_layer import PageText, TextLayerError

"""
Tests for how the Tesseract operations report failures, with pdf2image and tesserocr replaced by fakes, so they
also run where poppler and Tesseract aren't installed.
"""


def _fake_ocr(monkeypatch, pages: int, text: str, confidence: int) -> None:
    """
    Replaces pdf2image and tesserocr so every page returns the given text and confidence.
    """
    api = mock.MagicMock()
    api.GetUTF8Text.return_value = text
    api.MeanTextConf.return_value = confidence
    # As real Tesseract does when it can't tell a page's orientation, e.g. because the page is blank.
    api.DetectOrientationScript.return_value = None
    tess_api = mock.MagicMock()
    tess_api.return_value.__enter__.return_value = api
    monkeypatch.setattr(tesseract_operations.pdf2image, 'pdfinfo_from_path', lambda path: {'Pages': pages})
    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', lambda path, **kwargs: [mock.MagicMock()])
    monkeypatch.setattr(tesseract_operations.tesserocr, 'PyTessBaseAPI', tess_api)
    letter = {page: (8.5, 11.0) for page in range(1, pages + 1)}
    monkeypatch.setattr(tesseract_operations, 'page_sizes', lambda path, page_count: letter)


def test_doc_returns_average_confidence(monkeypatch):
    """
    Test that a document with text returns one result with its confidence.

    Asserts:
        The content and confidence come from the OCR output.
    """
    _fake_ocr(monkeypatch, pages=2, text='Patient is a good dog.', confidence=80)
    output = TesseractOperationDoc().process_row('good.pdf')
    assert len(output) == 1
    assert output[0]['confidence'] == 80
    assert 'good dog' in output[0]['content']


@pytest.mark.parametrize('operation', [TesseractOperationDoc, TesseractOperationPage])
def test_no_text_raises(monkeypatch, operation):
    """
    Test that a PDF with no recognizable text is reported as a failure, not a crash or an empty result.

    Asserts:
        NoTextFoundError naming the file is raised.
    """
    _fake_ocr(monkeypatch, pages=2, text='', confidence=0)
    with pytest.raises(NoTextFoundError, match='blank.pdf'):
        operation().process_row('blank.pdf')


def test_unreadable_text_layer_falls_back_to_ocr(monkeypatch, caplog):
    """
    Test that a PDF whose text layer poppler can't read is read with OCR, with a warning, rather than failing.

    Asserts:
        Every page comes from OCR, and the warning names the file and the problem.
    """
    _fake_ocr(monkeypatch, pages=2, text='Patient is a good dog.', confidence=80)

    def unreadable(path):
        raise TextLayerError('pdftotext exited with code 1 on odd.pdf: Syntax Error')

    monkeypatch.setattr(tesseract_operations, 'read_pages', unreadable)
    with caplog.at_level(logging.WARNING, logger='import_utility'):
        output = TesseractOperationPage().process_row('odd.pdf')
    assert [o['source'] for o in output] == [SOURCE_OCR, SOURCE_OCR]
    assert 'odd.pdf: its text layer could not be read, so every page is read with OCR' in caplog.text


def test_text_layer_pages_skip_ocr(monkeypatch, caplog):
    """
    Test that pages whose text layer is used are neither rendered nor read with OCR, and the rest are.

    Asserts:
        Page 1 comes from its text layer with no confidence, page 2 from OCR, and only page 2 is rendered.
        The document's confidence is page 2's, and its source is mixed.
        The log says how many pages came from each.
    """
    _fake_ocr(monkeypatch, pages=2, text='Recheck exam.', confidence=80)
    rendered = []
    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path',
                        lambda path, **kwargs: rendered.append(kwargs['first_page']) or [mock.MagicMock()])
    monkeypatch.setattr(tesseract_operations, 'read_pages', lambda path: [
        PageText(1, 'Annual exam. Weight 32 kg.', ''), PageText(2, '', 'no text layer')])
    with caplog.at_level(logging.INFO, logger='import_utility'):
        pages = TesseractOperationPage().process_row('mixed.pdf')
    assert [(o['source'], o['confidence']) for o in pages] == [(SOURCE_TEXT_LAYER, None), (SOURCE_OCR, 80)]
    assert pages[0]['content'] == 'Annual exam. Weight 32 kg.'
    assert rendered == [2]
    assert 'mixed.pdf: 1 of 2 pages read from the text layer, 1 with OCR (no text layer 1).' in caplog.text
    [document] = TesseractOperationDoc().process_row('mixed.pdf')
    assert (document['source'], document['confidence']) == (SOURCE_MIXED, 80)


def test_unreadable_pdf_raises(monkeypatch):
    """
    Test that an error reading the PDF reaches the caller instead of being swallowed.

    Asserts:
        The error is raised from process_row.
    """
    def broken(path):
        raise ValueError('Unable to get page count.')

    monkeypatch.setattr(tesseract_operations.pdf2image, 'pdfinfo_from_path', broken)
    with pytest.raises(ValueError, match='page count'):
        TesseractOperationDoc().process_row('broken.pdf')


@pytest.mark.parametrize('size, expected_dpi', [
    ((8.5, 11.0), 200),
    # The page that failed on real records: 188,817,300 pixels at 200 dpi.
    ((59.0, 80.0), 194),
    # A long, thin page: under the pixel limit, but 40,000 pixels long at 200 dpi, more than Tesseract reads.
    ((8.5, 200.0), 163),
    # pdfinfo didn't report the size, or reported a zero-width box.
    (None, 200),
    ((0.0, 1.4), 200),
])
def test_large_pages_render_at_lower_resolution(monkeypatch, size, expected_dpi):
    """
    Test that a page too large to render at 200 dpi within Pillow's size limit is rendered at a lower resolution,
    and a page of unknown size at 200 dpi.

    Asserts:
        Each page is rendered at the expected resolution, and its text is still read.
    """
    _fake_ocr(monkeypatch, pages=2, text='Patient is a good dog.', confidence=80)
    sizes = {1: size, 2: size} if size else {}
    monkeypatch.setattr(tesseract_operations, 'page_sizes', lambda path, page_count: sizes)
    rendered = []

    def convert(path, **kwargs):
        rendered.append((kwargs['first_page'], kwargs['dpi']))
        return [mock.MagicMock()]

    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', convert)
    output = TesseractOperationDoc().process_row('large.pdf')
    assert rendered == [(1, expected_dpi), (2, expected_dpi)]
    assert 'good dog' in output[0]['content']


def test_page_too_large_to_read_raises(monkeypatch):
    """
    Test that a page that would have to be rendered below MIN_RENDER_DPI fails as too large, not as having no text.

    Asserts:
        PageTooLargeError is raised, and the page isn't rendered.
    """
    _fake_ocr(monkeypatch, pages=1, text='', confidence=0)
    monkeypatch.setattr(tesseract_operations, 'page_sizes', lambda path, page_count: {1: (500.0, 500.0)})
    convert = mock.MagicMock()
    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', convert)
    with pytest.raises(PageTooLargeError, match='huge.pdf is 500 x 500 inches'):
        TesseractOperationDoc().process_row('huge.pdf')
    convert.assert_not_called()


def test_page_sizes_logs_when_pdfinfo_fails(monkeypatch, caplog):
    """
    Test that a pdfinfo -box failure is logged once, with what pdfinfo printed, instead of only showing up as
    every page's size being unknown.

    Asserts:
        No sizes are returned, and the warning names the exit code and pdfinfo's error.
    """
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr='Syntax Error: Couldn\'t read xref table\n')
    monkeypatch.setattr(tesseract_operations.subprocess, 'run', lambda *args, **kwargs: failed)
    with caplog.at_level(logging.WARNING, logger='import_utility'):
        assert tesseract_operations.page_sizes('damaged.pdf', 3) == {}
    assert "exited with code 1" in caplog.text
    assert "Couldn't read xref table" in caplog.text


def _fake_setup(monkeypatch, tools=('pdfinfo', 'pdftoppm', 'pdftotext', 'pdfimages'), tessdata_exists=True, models=('eng', 'osd'),
                init_error=None):
    """
    Replaces the poppler lookup, the tessdata directory and model checks, and Tesseract start-up.
    """
    tess_api = mock.MagicMock()
    if init_error:
        tess_api.side_effect = init_error
    monkeypatch.setattr(tesseract_operations.shutil, 'which', lambda tool: f'/usr/bin/{tool}' if tool in tools else None)
    monkeypatch.setattr(tesseract_operations.os.path, 'isdir', lambda path: tessdata_exists)
    monkeypatch.setattr(tesseract_operations.os.path, 'isfile',
                        lambda path: any(path.endswith(f'/{model}.traineddata') for model in models))
    monkeypatch.setattr(tesseract_operations.tesserocr, 'PyTessBaseAPI', tess_api)


def test_check_ocr_setup_passes_when_tools_are_ready(monkeypatch):
    """
    Test that a working setup passes the check.

    Asserts:
        No error is raised.
    """
    _fake_setup(monkeypatch)
    check_ocr_setup()


def test_check_ocr_setup_missing_poppler(monkeypatch):
    """
    Test that a missing poppler install stops the run with a clear error.

    Asserts:
        OcrSetupError names poppler and the missing tool.
    """
    _fake_setup(monkeypatch, tools=('pdftoppm',))
    with pytest.raises(OcrSetupError, match='Poppler is not installed.*pdfinfo'):
        check_ocr_setup()


def test_check_ocr_setup_missing_text_layer_tools(monkeypatch):
    """
    Test that poppler without its text tools stops the run, since the text layer is read with them.

    Asserts:
        OcrSetupError names the missing tools.
    """
    _fake_setup(monkeypatch, tools=('pdfinfo', 'pdftoppm'))
    with pytest.raises(OcrSetupError, match='missing pdftotext, pdfimages'):
        check_ocr_setup()


def test_check_ocr_setup_ocr_only_needs_no_text_layer_tools(monkeypatch):
    """
    Test that with --ocr-only, which never reads the text layer, poppler without its text tools is fine.

    Asserts:
        No error is raised.
    """
    _fake_setup(monkeypatch, tools=('pdfinfo', 'pdftoppm'))
    check_ocr_setup(text_layer=False)


def test_check_ocr_setup_missing_tessdata(monkeypatch):
    """
    Test that a missing tessdata directory stops the run with a clear error.

    Asserts:
        OcrSetupError names the missing directory.
    """
    _fake_setup(monkeypatch, tessdata_exists=False)
    with pytest.raises(OcrSetupError, match='/nowhere/tessdata does not exist'):
        check_ocr_setup('/nowhere/tessdata')


def test_check_ocr_setup_missing_osd_model(monkeypatch):
    """
    Test that a missing orientation model stops the run, since without it rotated pages are read as garbage.

    Asserts:
        OcrSetupError names the missing model.
    """
    _fake_setup(monkeypatch, models=('eng',))
    with pytest.raises(OcrSetupError, match='osd missing from /usr/share/tessdata'):
        check_ocr_setup()


def test_check_ocr_setup_unreadable_osd_model(monkeypatch):
    """
    Test that an osd model Tesseract can't load (e.g. a truncated download) stops the run, since the default start-up
    doesn't load it.

    Asserts:
        OcrSetupError names the osd model.
    """
    def tess_api(path, lang='eng', **kwargs):
        if lang == 'osd':
            raise RuntimeError('Failed to init API, possibly an invalid tessdata path')
        return mock.MagicMock()

    _fake_setup(monkeypatch)
    monkeypatch.setattr(tesseract_operations.tesserocr, 'PyTessBaseAPI', tess_api)
    with pytest.raises(OcrSetupError, match='could not load the osd model'):
        check_ocr_setup()


def test_check_ocr_setup_invalid_tessdata(monkeypatch):
    """
    Test that tessdata Tesseract can't start with (e.g. no language files) stops the run.

    Asserts:
        OcrSetupError includes Tesseract's own message.
    """
    _fake_setup(monkeypatch, init_error=RuntimeError('Failed to init API, possibly an invalid tessdata path'))
    with pytest.raises(OcrSetupError, match='could not start.*invalid tessdata path'):
        check_ocr_setup()
