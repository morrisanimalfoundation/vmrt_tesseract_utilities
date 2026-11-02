"""
Tests for the `pii_scrubber` script, run against the real Presidio analyzer and Hugging Face model.

A package update (presidio, transformers, spaCy, torch) can make the scrubber miss personal information without
anything crashing, so these tests check what it removes, what it keeps, and how much it finds. They need the
scrubber's packages, which are only installed in the workspace image, so they are skipped elsewhere. In CI they
run inside the image, where skipped tests fail the build. The first run downloads the model (about 400 MB).

All of the data below is made up.
"""

import argparse
import os
import sys
from collections import Counter

import pytest

# Everything the scrubber's NLP engine loads, including the spaCy model it uses for tokenizing, so a partial local
# install skips these tests instead of erroring.
for module in ('presidio_analyzer', 'torch', 'transformers', 'spacy_huggingface_pipelines', 'en_core_web_sm'):
    pytest.importorskip(module, reason='The PII scrubber packages are only installed in the workspace image.')

from scripts.scrubbers import pii_scrubber  # noqa: E402

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
    '012345',  # The subject ID's digits, so this also covers 094-012345 and 094012345.
    'Margaret',  # Also covers the email address, margaret.thompson@example.com.
    'Thompson',
    'example.com',
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

# Detections per entity type in RECORD, recorded with the package versions in requirements.txt and revision
# 661b9c1c717d3165512d440abc3700c386aefab6 of the StanfordAIMI/stanford-deidentifier-base model, which the image
# doesn't pin yet: a new model revision on Hugging Face can change these. Low-scoring types that only fire on parts
# of other entities (URL inside the email address, US_SSN and US_DRIVER_LICENSE on the subject ID) are left out
# because they are noise, not what we rely on.
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
    'GrlsSubjectIdRecognizer',
]

# A threshold above the 0.4 that the NLP config scales the model's ID detections down to.
RAISED_THRESHOLD = 0.5


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
def analyzer():
    """
    Creates the analyzer once for the whole module, with the scrubber's default config.

    Returns
    -------
    AnalyzerEngine
        The Presidio analyzer engine.
    """
    return pii_scrubber.create_nlp_engine(CONFIG_FILE)


@pytest.fixture(scope='module')
def scrubbed(analyzer):
    """
    Scrubs RECORD once for the whole module, the same way pii_scrubber.py does by default.

    Returns
    -------
    tuple
        The scrubbed text and the list of Presidio results.
    """
    return pii_scrubber.scrub_pii(RECORD, analyzer, threshold=0.0)


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


@pytest.mark.parametrize('subject_id', ['094-012345', '094012345', '094-0012345', '0940012345',
                                        '094 012345', '094–012345', 'O94-012345'])
def test_subject_id_is_removed_at_a_raised_threshold(analyzer, subject_id):
    """
    Test that a GRLS subject ID, in each form it takes in OCR'd text, is removed even with a raised threshold.
    """
    text, _ = pii_scrubber.scrub_pii(f'GRLS Subject ID: {subject_id}\n', analyzer, threshold=RAISED_THRESHOLD)
    assert '012345' not in text


@pytest.mark.parametrize('number', ['1094-012345', '094-01234567', '9094012345'])
def test_subject_id_pattern_ignores_longer_numbers(number):
    """
    Test that the subject ID pattern doesn't match inside a longer run of digits.
    """
    recognizer = pii_scrubber.create_subject_id_recognizer()
    assert recognizer.analyze(f'Ref {number} on file', entities=['ID']) == []


@pytest.mark.parametrize('until_done, chunks_scrubbed', [(False, 1), (True, 2)])
def test_scrub_chunks(monkeypatch, until_done, chunks_scrubbed):
    """
    Test that one chunk is scrubbed by default, and every chunk with --until-done.
    """
    chunks = [['first', 'second'], ['third'], []]
    scrubbed_chunks = []
    monkeypatch.setattr(pii_scrubber, 'get_files_to_process', lambda args: chunks[len(scrubbed_chunks)])
    monkeypatch.setattr(pii_scrubber, 'process_files',
                        lambda files, analyzer, output_dir, threshold: scrubbed_chunks.append(files))
    args = argparse.Namespace(output_to='output', threshold=0.0, until_done=until_done)

    pii_scrubber.scrub_chunks(args, analyzer=None)

    assert scrubbed_chunks == chunks[:chunks_scrubbed]


def test_offset_is_rejected_with_until_done(monkeypatch):
    """
    Test that --offset can't be combined with --until-done, which would leave that many files unscrubbed.
    """
    monkeypatch.setattr(sys, 'argv', ['pii_scrubber.py', 'output', '--until-done', '--offset', '10'])
    with pytest.raises(SystemExit):
        pii_scrubber.parse_args()
