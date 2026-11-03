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
import yaml
from PIL import Image, ImageDraw, ImageFont

from vmrt_tesseract_utilities.tesseract_operations import TesseractOperationDoc

EXPECTED_TEXT = 'Vaccines current'

SCRUBBER_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'scrubbers', 'config',
                               'stanford-deidentifier-base_nlp.yaml')


def check_imports() -> None:
    """The PII scrubber's NLP engine needs these, and they have gone missing before (#21)."""
    import presidio_analyzer  # noqa: F401
    import presidio_analyzer.nlp_engine.huggingface_token_pipe  # noqa: F401
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


def check_scrubber_model() -> None:
    """The PII scrubber's pinned model is in the folder its config points at, and nothing is fetched at run time."""
    with open(SCRUBBER_CONFIG) as f:
        model_dir = yaml.safe_load(f)['models'][0]['model_name']['transformers']
    for name in ('config.json', 'pytorch_model.bin', 'vocab.txt', 'REVISION'):
        if not os.path.isfile(os.path.join(model_dir, name)):
            raise AssertionError(f'The PII scrubber model is missing {name} in {model_dir}')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise AssertionError('HF_HUB_OFFLINE should be 1, so the scrubber never downloads a different model.')


def check_home_is_writable() -> None:
    """The container runs as a user with no account in the image, and tools (pip --user, caches) write to HOME."""
    with tempfile.NamedTemporaryFile(dir=os.environ['HOME']):
        pass


if __name__ == '__main__':
    check_imports()
    check_spacy_models()
    check_scrubber_model()
    check_home_is_writable()
    with tempfile.TemporaryDirectory() as tmp:
        check_ocr(tmp)
    print('Docker image smoke test passed.')
    sys.exit(0)
