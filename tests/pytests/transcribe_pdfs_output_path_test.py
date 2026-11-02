import pytest

"""
Tests for how `transcribe_pdfs` names its output files.

The script imports the Tesseract operations, which need Tesseract and poppler, so these are skipped outside the
Docker image.
"""

pytest.importorskip('tesserocr')
pytest.importorskip('pdf2image')

from scripts.transcribe_pdfs import _output_path  # noqa: E402
from vmrt_tesseract_utilities.database import TranscriptionInput  # noqa: E402


def test_same_file_name_in_different_folders_gets_different_paths():
    """
    Test that two PDFs with the same file name, in different dogs' folders, don't share an output file.
    """
    first = TranscriptionInput(id=1, input_file='/data/094-000001/Medical Records.pdf')
    second = TranscriptionInput(id=2, input_file='/data/094-000002/Medical Records.pdf')
    result = {'page': 3, 'block': 0}

    assert _output_path('out', first, result) == 'out/1-Medical Records-3-0.txt'
    assert _output_path('out', second, result) == 'out/2-Medical Records-3-0.txt'
