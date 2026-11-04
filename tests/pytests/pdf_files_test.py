import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from vmrt_tesseract_utilities.database import Base, TranscriptionInput
from vmrt_tesseract_utilities.pdf_files import is_pdf_column, is_pdf_path

"""
Tests for the rule registration and OCR share for which files are PDFs.
"""

CASES = [
    ('/data/094-000131/record.pdf', True),
    ('/data/094-000131/SCAN.PDF', True),
    ('/data/094-000131/2023_exam.Pdf', True),
    ('/data/094-000131/._record.pdf', False),
    ('/data/094-000131/._export/record.pdf', False),
    ('/data/094-000131/exam/LOCKFILE', False),
    ('/data/094-000131/image.jpg', False),
    ('/data/094-000131/record.pdf.bak', False),
    # "_" is a LIKE wildcard: an unescaped "._" pattern would also match ".x".
    ('/data/094-000131/.xrecord.pdf', True),
]


@pytest.mark.parametrize('path, expected', CASES)
def test_is_pdf_path(path, expected):
    """
    Test the rule on paths.

    Asserts:
        .pdf in any case counts, unless a part of the path starts with "._"; other files don't.
    """
    assert is_pdf_path(path) is expected


def test_is_pdf_column_matches_is_pdf_path():
    """
    Test that the query condition picks the same rows as is_pdf_path.

    Asserts:
        Querying with is_pdf_column returns exactly the paths is_pdf_path accepts.
    """
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(TranscriptionInput(document_type='document', input_file=path) for path, _ in CASES)
        session.commit()
        selected = {path for (path,) in session.query(TranscriptionInput.input_file)
                    .where(is_pdf_column(TranscriptionInput.input_file))}
    assert selected == {path for path, expected in CASES if expected}
