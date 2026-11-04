"""
Which files the pipeline treats as PDFs to transcribe.

Registration (create_transcription_process.py) and OCR (transcribe_pdfs.py) both use this rule, so they agree on
which files are in scope.

A file counts if its name ends in ".pdf" in any case, unless a part of its path starts with "._". macOS writes
"._<name>" files next to the real ones to hold extra file information; they end in ".pdf" too but aren't PDFs.
"""

from sqlalchemy import and_, func, not_
from sqlalchemy.sql.elements import ColumnElement

# AppleDouble files ("._Medical Records.pdf"), and anything inside a folder named that way.
_APPLE_DOUBLE = '/._'


def is_pdf_path(path: str) -> bool:
    """
    Tells whether a file is a PDF the pipeline should transcribe.

    Parameters
    ----------
    path: str
        The file's path, with "/" between folders.

    Returns
    -------
    bool
        True for a ".pdf" file (any case) with no part of its path starting with "._".

    Examples
    --------
    >>> is_pdf_path('/data/094-000131/Medical Records.PDF')
    True
    >>> is_pdf_path('/data/094-000131/._Medical Records.pdf')
    False
    >>> is_pdf_path('/data/094-000131/exam/LOCKFILE')
    False
    """
    return path.lower().endswith('.pdf') and _APPLE_DOUBLE not in '/' + path


def is_pdf_column(column: ColumnElement) -> ColumnElement:
    """
    The same rule as is_pdf_path, as a condition on a path column for a database query.

    Parameters
    ----------
    column: ColumnElement
        The column holding the path, e.g. TranscriptionInput.input_file. Paths are absolute, so every part of the
        path follows a "/".

    Returns
    -------
    ColumnElement
        The condition. input_file has a case-sensitive collation, so the extension is compared in lower case;
        "_" is a wildcard in LIKE, so the "._" pattern is escaped.
    """
    return and_(func.lower(column).like('%.pdf'), not_(column.contains(_APPLE_DOUBLE, autoescape=True)))
