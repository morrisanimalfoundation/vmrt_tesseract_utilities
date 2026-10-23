from unittest import mock

import pytest

from vmrt_tesseract_utilities import tesseract_operations
from vmrt_tesseract_utilities.tesseract_operations import (
    NoTextFoundError, TesseractOperationDoc, TesseractOperationPage)

"""
Tests for the Tesseract operations.
"""


def _fake_ocr(monkeypatch, pages: int, text: str, confidence: int) -> None:
    """
    Replaces pdf2image and tesserocr so every page returns the given text and confidence.
    """
    api = mock.MagicMock()
    api.GetUTF8Text.return_value = text
    api.MeanTextConf.return_value = confidence
    tess_api = mock.MagicMock()
    tess_api.return_value.__enter__.return_value = api
    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', lambda path: [object()] * pages)
    monkeypatch.setattr(tesseract_operations.tesserocr, 'PyTessBaseAPI', tess_api)


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


def test_unreadable_pdf_raises(monkeypatch):
    """
    Test that an error converting the PDF reaches the caller instead of being swallowed.

    Asserts:
        The conversion error is raised from process_row.
    """
    def broken(path):
        raise ValueError('Unable to get page count.')

    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', broken)
    with pytest.raises(ValueError, match='page count'):
        TesseractOperationDoc().process_row('broken.pdf')
