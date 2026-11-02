"""
Smoke test for the workspace Docker image.

Checks that the system tools, Tesseract models and Python packages the pipeline needs are installed and work,
by running the OCR code on a generated PDF. Run it inside the image with the repository mounted at /workspace,
as a user that has no account in the image (as docker-compose.yml does):

    docker run --rm --user 1234:1234 -v "$PWD:/workspace" <image> python tests/docker_smoke_test.py

It is not part of the pytest suite, which runs both without Docker and, in CI, inside the image.
"""

import os
import sys
import tempfile

import spacy
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image, ImageDraw, ImageFont

from vmrt_tesseract_utilities.tesseract_operations import TesseractOperationDoc

EXPECTED_TEXT = 'Vaccines current'


def check_imports() -> None:
    """The PII scrubber's NLP engine needs these, and they have gone missing before (#21)."""
    import presidio_analyzer  # noqa: F401
    import spacy_huggingface_pipelines  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401


def check_spacy_models() -> None:
    """Both spaCy models are downloaded at build time."""
    for model in ('en_core_web_sm', 'en_core_web_lg'):
        spacy.load(model)


def check_ocr(workdir: str) -> None:
    """Poppler, Tesseract and the English model work together on a generated PDF."""
    image = Image.new('L', (1700, 600), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    ImageDraw.Draw(image).text((100, 250), f'{EXPECTED_TEXT}, weight 32 kg.', fill='black', font=font)
    pdf_path = os.path.join(workdir, 'smoke.pdf')
    image.save(pdf_path, 'PDF', resolution=200)

    output = TesseractOperationDoc().process_row(pdf_path)
    if not output or EXPECTED_TEXT not in output[0]['content']:
        raise AssertionError(f'OCR did not find {EXPECTED_TEXT!r}; got {output!r}')


def check_model_cache_is_writable() -> None:
    """The PII scrubber downloads its Hugging Face model into this cache on first use."""
    os.makedirs(HF_HUB_CACHE, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=HF_HUB_CACHE):
        pass


if __name__ == '__main__':
    check_imports()
    check_spacy_models()
    check_model_cache_is_writable()
    with tempfile.TemporaryDirectory() as tmp:
        check_ocr(tmp)
    print('Docker image smoke test passed.')
    sys.exit(0)
