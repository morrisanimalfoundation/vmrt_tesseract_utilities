import importlib
import sys
from unittest import mock

"""
Shared pytest setup.

The OCR libraries need system packages (poppler, tesseract) that aren't installed in CI,
so stand-ins are registered when they can't be imported. Tests patch the attributes they use.
"""

for module_name in ('pdf2image', 'tesserocr'):
    try:
        importlib.import_module(module_name)
    except ImportError:
        sys.modules[module_name] = mock.MagicMock()
