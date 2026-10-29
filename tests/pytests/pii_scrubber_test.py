"""
Tests for the `pii_scrubber` script, run against the real Presidio analyzer and Hugging Face model.

A package update (presidio, transformers, spaCy, torch) can make the scrubber miss personal information without
anything crashing, so these tests check what it removes, what it keeps, and how much it finds. They need the
scrubber's packages, which are only installed in the workspace image, so they are skipped elsewhere. In CI they
run inside the image, where skipped tests fail the build. The first run downloads the model (about 400 MB).

All of the data below is made up.
"""

import os
from collections import Counter

import pytest

pytest.importorskip('presidio_analyzer', reason='The PII scrubber packages are only installed in the workspace image.')

from scripts.scrubbers.pii_scrubber import (create_nlp_engine,  # noqa: E402
                                            scrub_pii)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'scrubbers', 'config',
                           'stanford-deidentifier-base_nlp.yaml')

RECORD = """\
Sunny Meadows Animal Hospital
1428 Larkspur Lane, Springfield, OR 97477
Phone: (541) 555-0142   Fax: (541) 555-0187

Patient: Biscuit   Species: Canine   Breed: Golden Retriever   Sex: FS
GRLS Subject ID: 094-012345
Owner: Margaret Thompson   Phone: 541-555-0163   Email: margaret.thompson@example.com
Attending veterinarian: Dr. Alan Whitaker, DVM

Visit date: March 14, 2023
History: Annual wellness exam. Owner reports good appetite and normal energy.
Exam: Weight 31.2 kg. Temperature 101.4 F. Patient is bright, alert, responsive.
Mild dental tartar noted on upper premolars. Heart and lungs auscultate normally.
Assessment: Healthy adult dog.
Plan: Rabies and DHPP vaccines given. Recheck in 6 months.
Next appointment: 09/14/2023 with Dr. Whitaker.
Signed, Jennifer Ortiz, LVT
"""

# Every identifier planted in RECORD. The dog's name is not here: replace_strings.py removes dog names earlier in
# the pipeline, from a list of known names.
IDENTIFIERS = [
    'Sunny Meadows',
    '1428',
    'Larkspur',
    'Springfield',
    '97477',
    '555-0142',
    '555-0187',
    '555-0163',
    '094-012345',
    '012345',
    'Margaret',
    'Thompson',
    'margaret.thompson@example.com',
    'Alan',
    'Whitaker',
    'Jennifer',
    'Ortiz',
    'March 14, 2023',
    '09/14/2023',
]

# Ordinary medical text that must survive, to catch over-scrubbing.
MEDICAL_TEXT = [
    'Golden Retriever',
    'Annual wellness exam',
    'Weight 31.2 kg',
    'Temperature 101.4 F',
    'bright, alert, responsive',
    'Mild dental tartar noted on upper premolars',
    'Heart and lungs auscultate normally',
    'Healthy adult dog',
    'Rabies and DHPP vaccines given',
    'Recheck in 6 months',
]

# Detections per entity type in RECORD, recorded with the package and model versions in the image. Low-scoring
# types that only fire on parts of other entities (URL inside the email address, US_SSN and US_DRIVER_LICENSE on
# the subject ID) are left out because they are noise, not what we rely on.
BASELINE_COUNTS = {
    'PERSON': 6,
    'PHONE_NUMBER': 5,
    'DATE_TIME': 2,
    'LOCATION': 2,
    'ID': 2,
    'EMAIL_ADDRESS': 1,
}

# Recognizers that each find something in RECORD. The transformer model finds most entities, often overlapping
# Presidio's pattern recognizers, so losing one of these may not change the counts above but must still fail.
EXPECTED_RECOGNIZERS = [
    'TransformersRecognizer',
    'PhoneRecognizer',
    'EmailRecognizer',
]


def allowed_difference(baseline: int) -> int:
    """
    How far a count may drift from its baseline: a quarter of it, rounded down, so small counts must match exactly.

    >>> allowed_difference(6)
    1
    >>> allowed_difference(2)
    0
    """
    return baseline // 4


@pytest.fixture(scope='module')
def scrubbed():
    """
    Scrubs RECORD once for the whole module, the same way pii_scrubber.py does by default.

    Returns
    -------
    tuple
        The scrubbed text and the list of Presidio results.
    """
    analyzer = create_nlp_engine(CONFIG_FILE)
    return scrub_pii(RECORD, analyzer, threshold=0.0)


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_identifier_is_removed(scrubbed, identifier):
    """
    Test that each planted identifier is gone from the scrubbed text.
    """
    text, _ = scrubbed
    assert identifier.lower() not in text.lower()


@pytest.mark.parametrize('medical_text', MEDICAL_TEXT)
def test_medical_text_survives(scrubbed, medical_text):
    """
    Test that ordinary medical text is not scrubbed.
    """
    text, _ = scrubbed
    assert medical_text in text


@pytest.mark.parametrize('entity_type, baseline', BASELINE_COUNTS.items())
def test_detection_count_matches_baseline(scrubbed, entity_type, baseline):
    """
    Test that the number of detections of each entity type stays close to its baseline.

    A big drop means an update made the scrubber miss things; a big rise means it started flagging ordinary text.
    If a change is expected (for example after deliberately updating the model), update BASELINE_COUNTS.
    """
    _, results = scrubbed
    count = Counter(result.entity_type for result in results)[entity_type]
    assert abs(count - baseline) <= allowed_difference(baseline), \
        f'Found {count} {entity_type} entities, expected {baseline} ± {allowed_difference(baseline)}'


@pytest.mark.parametrize('recognizer', EXPECTED_RECOGNIZERS)
def test_recognizer_still_finds_entities(scrubbed, recognizer):
    """
    Test that each recognizer we rely on still contributes at least one detection.
    """
    _, results = scrubbed
    found = {result.recognition_metadata['recognizer_name'] for result in results}
    assert recognizer in found, f'{recognizer} found nothing; recognizers that did: {sorted(found)}'
