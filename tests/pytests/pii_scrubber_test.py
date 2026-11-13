"""
Tests for the `pii_scrubber` script, run against the real Presidio analyzer and Hugging Face model.

A package update (presidio, transformers, spaCy, torch) can make the scrubber miss personal information without
anything crashing, so these tests check what it removes, what it keeps, and how much it finds. They need the
scrubber's packages, which are only installed in the workspace image, so they are skipped elsewhere. In CI they
run inside the image, where skipped tests fail the build. The model is the pinned copy in the image, so no
internet connection is needed.

All of the data below is made up.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

import pytest

# Everything the scrubber's NLP engine loads, including the spaCy model it uses for tokenizing, so a partial local
# install skips these tests instead of erroring.
for module in ('presidio_analyzer', 'presidio_analyzer.nlp_engine.huggingface_token_pipe', 'torch', 'transformers',
               'en_core_web_sm'):
    pytest.importorskip(module, reason='The PII scrubber packages are only installed in the workspace image.')

import yaml  # noqa: E402
from presidio_analyzer import RecognizerResult  # noqa: E402

from scripts.scrubbers import pii_scrubber  # noqa: E402
from vmrt_tesseract_utilities.date_extractor import DateExtractor  # noqa: E402
from vmrt_tesseract_utilities.scrubbing_utils import split_text  # noqa: E402

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'scrubbers', 'config',
                           'stanford-deidentifier-base_nlp.yaml')

# The model is built into the image, so it is missing anywhere else, even where the packages are installed.
if pii_scrubber.find_missing_model_folders(CONFIG_FILE):
    pytest.skip('The PII scrubber model is only in the workspace image.', allow_module_level=True)

# The model the baseline below was recorded with, as the image's Dockerfile writes it to the model folder's REVISION
# file. If the pin is updated, check the scrubbing quality and these counts before changing this.
BASELINE_MODEL = 'StanfordAIMI/stanford-deidentifier-base@661b9c1c717d3165512d440abc3700c386aefab6'

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
    'Larkspur Lane',
    'Springfield',
    'OR 97477',
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
]

# Dates in RECORD, which are left in the text and listed in the confidence report for the visit date miner.
DATES = ['March 14, 2023', '09/14/2023']

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

# Detections per entity type in RECORD, recorded with the package versions in requirements.txt and BASELINE_MODEL.
# Updating the model's pin can change these. Low-scoring types that only fire on parts
# of other entities (US_SSN and US_DRIVER_LICENSE on the subject ID) are left out because they are noise, not what
# we rely on. The model's detections inside the clinic's address are merged into the ADDRESS.
BASELINE_COUNTS = {
    'PERSON': 5,
    'PHONE_NUMBER': 4,
    'DATE_TIME': 2,
    'LOCATION': 1,
    'ADDRESS': 1,
    'ID': 1,
    'EMAIL_ADDRESS': 1,
}

# Recognizers that each find something in RECORD. The transformer model finds most entities, often overlapping
# Presidio's pattern recognizers, so losing one of these may not change the counts above but must still fail.
EXPECTED_RECOGNIZERS = [
    'TransformersRecognizer',
    'PhoneRecognizer',
    'EmailRecognizer',
    'GrlsSubjectIdRecognizer',
    'UsAddressRecognizer',
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


@pytest.mark.parametrize('date', DATES)
def test_dates_are_kept(scrubbed, date):
    """
    Test that dates are left in the scrubbed text, and listed in its results as kept.
    """
    text, results = scrubbed
    assert date in text
    assert date in [RECORD[result.start:result.end] for result in results if result.entity_type == 'DATE_TIME']


def test_confidence_report_lists_dates_for_the_visit_date_miner(scrubbed, tmp_path):
    """
    Test that the confidence report still lists dates, marked as not scrubbed, so the visit date miner finds them.
    """
    _, results = scrubbed
    report = tmp_path / 'confidence.json'

    pii_scrubber.write_confidence_record(str(report), results, RECORD)

    entries = json.loads(report.read_text())
    assert {entry['Original Text'] for entry in entries if not entry['Scrubbed']} == set(DATES)
    assert all(entry['Scrubbed'] for entry in entries if entry['Type'] != 'DATE_TIME')
    found = DateExtractor(str(report)).extract_dates_from_file()
    assert datetime(2023, 3, 14) in found and datetime(2023, 9, 14) in found


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
                                        '094 012345', '094–012345', 'O94-012345', '094-\n012345',
                                        '094-\n\n012345'])
def test_subject_id_is_removed_at_a_raised_threshold(analyzer, subject_id):
    """
    Test that a GRLS subject ID, in each form it takes in OCR'd text, is removed even with a raised threshold.
    """
    text, _ = pii_scrubber.scrub_pii(f'GRLS Subject ID: {subject_id}\n', analyzer, threshold=RAISED_THRESHOLD)
    assert '012345' not in text


def test_split_subject_id_keeps_its_line_breaks(analyzer):
    """
    Test that a subject ID split across line breaks is replaced whole, without joining the lines around it.
    """
    text = 'Patient ID 094-\n\n012345 seen for a recheck.\nWeight 31.2 kg\n'

    scrubbed, _ = pii_scrubber.scrub_pii(text, analyzer, threshold=RAISED_THRESHOLD)

    assert '094' not in scrubbed and '012345' not in scrubbed
    assert '<ID>\n\n' in scrubbed
    assert scrubbed.count('\n') == text.count('\n')


@pytest.mark.parametrize('number', ['1094-012345', '094-01234567', '9094012345'])
def test_subject_id_pattern_ignores_longer_numbers(number):
    """
    Test that the subject ID pattern doesn't match inside a longer run of digits.
    """
    recognizer = pii_scrubber.create_subject_id_recognizer()
    assert recognizer.analyze(f'Ref {number} on file', entities=['ID']) == []


def test_long_text_is_analyzed_in_pieces(analyzer):
    """
    Test that text analyzed in pieces finds the same entities, at the right places, as each piece would alone.

    Each copy of RECORD is one piece, since RECORD ends with a line break, so every copy's results should match
    RECORD's, shifted by the copy's offset.
    """
    _, whole = pii_scrubber.scrub_pii(RECORD, analyzer, threshold=0.0)
    expected = sorted((result.entity_type, result.start, result.end) for result in whole)

    copies = 3
    text, results = pii_scrubber.scrub_pii(RECORD * copies, analyzer, threshold=0.0, max_length=len(RECORD))

    for copy in range(copies):
        offset = copy * len(RECORD)
        found = sorted((result.entity_type, result.start - offset, result.end - offset) for result in results
                       if offset <= result.start < offset + len(RECORD))
        assert found == expected, f'copy {copy}'
    for identifier in IDENTIFIERS:
        assert identifier.lower() not in text.lower()


def test_subject_id_is_not_split_between_pieces(analyzer):
    """
    Test that a long text isn't split into pieces in the middle of a subject ID.

    The piece length makes the first piece end at the space inside "094 012345", so the ID is only found, and
    scrubbed, if the piece ends before it instead.
    """
    text = 'Seen today for a recheck, ID 094 012345 on file.\n'
    max_length = text.index(' 012345') + 1

    scrubbed, results = pii_scrubber.scrub_pii(text, analyzer, threshold=0.5, max_length=max_length)

    start = text.index('094')
    assert ('ID', start, start + len('094 012345')) in [(r.entity_type, r.start, r.end) for r in results]
    assert '012345' not in scrubbed


def test_offset_is_rejected_with_until_done(monkeypatch):
    """
    Test that --offset can't be combined with --until-done, which would leave that many files unscrubbed.
    """
    monkeypatch.setattr(sys, 'argv', ['pii_scrubber.py', 'output', '--until-done', '--offset', '10'])
    with pytest.raises(SystemExit):
        pii_scrubber.parse_args()


def test_model_is_the_baseline_revision():
    """
    Test that the image's model is the revision the baseline counts were recorded with.
    """
    with open(CONFIG_FILE) as f:
        model_dir = yaml.safe_load(f)['models'][0]['model_name']['transformers']
    with open(os.path.join(model_dir, 'REVISION')) as f:
        assert f.read().strip() == BASELINE_MODEL


def test_missing_model_folder_is_reported(tmp_path):
    """
    Test that a config naming a model folder that doesn't exist fails with a clear error, as it would in an image
    built before the model was added.
    """
    missing = str(tmp_path / 'no-such-model')
    config_file = tmp_path / 'nlp.yaml'
    with open(CONFIG_FILE) as f:
        config_file.write_text(f.read().replace('/opt/models/stanford-deidentifier-base', missing))

    with pytest.raises(FileNotFoundError, match='no-such-model'):
        pii_scrubber.create_nlp_engine(str(config_file))


@pytest.mark.parametrize('text, clinic, address', [
    ('Clinic: Cedar Ridge Pet Hospital, 77 Old Mill Rd, Bend OR 97701\n', 'Cedar Ridge',
     '77 Old Mill Rd, Bend OR 97701'),
    ('Bayview Veterinary Clinic, 220 W. Harbor Blvd Suite 4, Ventura, California 93001\n', 'Bayview',
     '220 W. Harbor Blvd Suite 4, Ventura, California 93001'),
    ('Sunny Meadows Animal Hospital\n1428 Larkspur Lane\nSpringfield, OR 97477\nPhone: (541) 555-0142\n',
     'Sunny Meadows', '1428 Larkspur Lane\nSpringfield, OR 97477'),
])
def test_whole_clinic_address_is_scrubbed(analyzer, text, clinic, address):
    """
    Test that a clinic's name and whole address (street, city, state and ZIP) are scrubbed, the address as one,
    keeping the line breaks.
    """
    scrubbed, _ = pii_scrubber.scrub_pii(text, analyzer, threshold=0.0)

    assert '<ADDRESS>' in scrubbed
    for part in [clinic] + address.replace(',', ' ').split():
        assert part not in scrubbed
    assert scrubbed.count('\n') == text.count('\n')


@pytest.mark.parametrize('text', [
    'Pt is lethargic.do not vaccinate today.\n',
    'Owner reports dog is quiet.today.\n',
    'Owner called. vomiting.no diarrhea\n',
])
def test_words_joined_by_a_full_stop_are_kept(analyzer, text):
    """
    Test that words OCR joined with a full stop aren't scrubbed as URLs.
    """
    scrubbed, results = pii_scrubber.scrub_pii(text, analyzer, threshold=0.0)

    assert scrubbed == text
    assert 'URL' not in [result.entity_type for result in results]


@pytest.mark.parametrize('url', [
    'https://pacs.sunnymeadowsvet.com/study/8812',
    'http://192.168.1.20/xray/44',
    'www.vetimaging.net/x/1',
    'sunnymeadowsvet.com/xray',
])
def test_urls_are_scrubbed(analyzer, url):
    """
    Test that real URLs, such as links to a clinic's own x-ray server, are still scrubbed.
    """
    scrubbed, _ = pii_scrubber.scrub_pii(f'Radiographs: {url} reviewed.\n', analyzer, threshold=0.0)

    assert url not in scrubbed
    assert 'Radiographs:' in scrubbed and 'reviewed.' in scrubbed


STAFF_LOG = """\
Date     Staff  Notes
03/02/21 JO     Exam by Jennifer. Ortiz to call owner.
03/04/21 AW     Spoke with Mrs. Thompson re: labs. Dr Whitaker reviewed.
03/09/21 KB     JENNIFER ORTIZ LVT. Owner: MARGARET THOMPSON
03/10/21 J.O.   Signed: alan whitaker DVM
"""


@pytest.mark.parametrize('initials', ['JO ', 'AW ', 'KB ', 'J.O. '])
def test_staff_initials_are_kept(analyzer, initials):
    """
    Test that staff initials are kept.
    """
    scrubbed, _ = pii_scrubber.scrub_pii(STAFF_LOG, analyzer, threshold=0.0)
    assert initials in scrubbed


@pytest.mark.parametrize('name', ['Jennifer', 'Ortiz', 'Thompson', 'Whitaker', 'JENNIFER', 'MARGARET', 'alan'])
def test_names_are_scrubbed(analyzer, name):
    """
    Test that first and last names are scrubbed, whatever their case, alongside initials that are kept.
    """
    scrubbed, _ = pii_scrubber.scrub_pii(STAFF_LOG, analyzer, threshold=0.0)
    assert name.lower() not in scrubbed.lower()


def make_result(entity_type, start, end, score=0.5):
    """
    Makes a detection for the merge tests.
    """
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)


def test_overlapping_detections_are_merged_into_an_address():
    """
    Test that detections overlapping an address, directly or through each other, become part of it.
    """
    address = make_result('ADDRESS', 10, 30, 0.95)
    person = make_result('PERSON', 5, 15, 1.0)  # Starts before the address.
    zip_id = make_result('ID', 25, 35)  # Ends after it.
    phone = make_result('PHONE_NUMBER', 33, 40)  # Only overlaps the ID.
    separate = make_result('PERSON', 50, 55)

    merged = pii_scrubber.merge_into_addresses([separate, zip_id, person, address, phone])

    assert [(r.entity_type, r.start, r.end) for r in merged] == [('ADDRESS', 5, 40), ('PERSON', 50, 55)]


def test_overlaps_without_an_address_are_left_alone():
    """
    Test that overlapping detections with no address among them aren't merged.
    """
    results = [make_result('ORGANIZATION', 0, 20), make_result('URL', 0, 20), make_result('PERSON', 30, 40)]

    merged = pii_scrubber.merge_into_addresses(results)

    assert [(r.entity_type, r.start, r.end) for r in merged] == [('ORGANIZATION', 0, 20), ('URL', 0, 20),
                                                                 ('PERSON', 30, 40)]


class StubAnalyzer:
    """
    Stands in for the analyzer, returning fixed detections, to test how overlaps are handled exactly.
    """

    def __init__(self, detections):
        """
        Parameters
        ----------
        detections : list[tuple]
            (entity type, start, end) or (entity type, start, end, recognizer name) for each detection.
        """
        self.detections = detections

    def analyze(self, text, language):
        """
        Returns new copies of the detections, since scrub_pii shifts their offsets.
        """
        return [RecognizerResult(entity_type=d[0], start=d[1], end=d[2], score=0.9,
                                 recognition_metadata={'recognizer_name': d[3] if len(d) > 3 else 'Stub'})
                for d in self.detections]


def span(text, part, entity_type, recognizer=None):
    """
    Makes a stub detection of part, the first time it appears in text.
    """
    start = text.index(part)
    return (entity_type, start, start + len(part)) + ((recognizer,) if recognizer else ())


def scrub_with_report(text, detections, tmp_path):
    """
    Scrubs text with the stub analyzer and writes its confidence report.

    Returns
    -------
    tuple
        The scrubbed text and {(type, original text): scrubbed} from the report.
    """
    scrubbed, results = pii_scrubber.scrub_pii(text, StubAnalyzer(detections), threshold=0.0)
    report = tmp_path / 'confidence.json'
    pii_scrubber.write_confidence_record(str(report), results, text)
    return scrubbed, {(entry['Type'], entry['Original Text']): entry['Scrubbed']
                      for entry in json.loads(report.read_text())}


@pytest.mark.parametrize('text, found, entity_type, expected', [
    # The shapes of what the first real-record QA check found left next to a placeholder.
    ('Clinic\n1234 N Cedar Ridge STE 1000\nNext', 'Cedar Ridge', 'LOCATION', 'Clinic\n<ADDRESS>\nNext'),
    ('Owner: Ann Lee, Springfield 97477', 'Springfield', 'LOCATION', 'Owner: Ann Lee, <ADDRESS>'),
    ('Home: Maple Grove Ct (541) 555-0134', 'Grove', 'LOCATION', 'Home: <ADDRESS> (541) 555-0134'),
    ('PO Box 112 Two Rivers Rd', 'PO Box 112', 'ADDRESS', '<ADDRESS>'),
    ('Clinic: 500 Main St; Suite 12', '500 Main St', 'ADDRESS', 'Clinic: <ADDRESS>'),
    # Nothing next to it is part of an address.
    ('Referred to Bend CT scan', 'Bend', 'LOCATION', 'Referred to <LOCATION> CT scan'),
])
def test_address_parts_next_to_a_place_are_scrubbed(text, found, entity_type, expected):
    """
    Test that a house number, suite, ZIP code or the rest of a street next to a place or address the model found is
    scrubbed with it.
    """
    scrubbed, _ = pii_scrubber.scrub_pii(text, StubAnalyzer([span(text, found, entity_type)]), threshold=0.0)

    assert scrubbed == expected


def test_address_is_not_stretched_over_a_kept_date(tmp_path):
    """
    Test that a number before a place isn't taken for a house number when it's part of a kept date.
    """
    text = 'Seen June 3 Bend OR 97701'

    scrubbed, report = scrub_with_report(text, [span(text, 'June 3', 'DATE_TIME'), span(text, 'Bend', 'LOCATION')],
                                         tmp_path)

    assert scrubbed == 'Seen June 3 <ADDRESS>'
    assert report == {('DATE_TIME', 'June 3'): False, ('ADDRESS', 'Bend OR 97701'): True}


def test_letterhead_address_parts_are_scrubbed(analyzer):
    """
    Test that a letterhead's house number, suite and ZIP code are scrubbed with the street and city the model finds,
    even when the street has no street type and the city no state.
    """
    text = 'Cedar Ridge Animal Clinic\n1234 N Juniper Ridge STE 1000\nSpringfield 97477\nPhone (541) 555-0134\n'

    scrubbed, _ = pii_scrubber.scrub_pii(text, analyzer, threshold=0.5)

    for part in ('1234', 'Juniper', '1000', 'Springfield', '97477'):
        assert part not in scrubbed


def test_name_inside_a_date_is_kept(tmp_path):
    """
    Test that a detection inside a kept date, such as "June" read as a name, doesn't cut the date apart.
    """
    text = 'Seen June 3, 2023 by Jennifer Ortiz.\n'
    detections = [span(text, 'June 3, 2023', 'DATE_TIME'), span(text, 'June', 'PERSON'),
                  span(text, 'Jennifer Ortiz', 'PERSON')]

    scrubbed, report = scrub_with_report(text, detections, tmp_path)

    assert scrubbed == 'Seen June 3, 2023 by <PERSON>.\n'
    # Still listed in the report, so QA can see what was kept with a date.
    assert report == {('DATE_TIME', 'June 3, 2023'): False, ('PERSON', 'June'): False,
                      ('PERSON', 'Jennifer Ortiz'): True}


@pytest.mark.parametrize('text, date, inside, entity_type', [
    # Pattern recognizers that read a numeric date as a phone, license or account number.
    ('Seen 03-14-23 for vaccines.\n', '03-14-23', '03-14-23', 'PHONE_NUMBER'),
    ('Seen 03-14-2023 for vaccines.\n', '03-14-2023', '03-14-2023', 'PHONE_NUMBER'),
    ('Invoice date 031423.\n', '031423', '031423', 'US_DRIVER_LICENSE'),
    ('Invoice date 20230314.\n', '20230314', '20230314', 'US_BANK_NUMBER'),
])
def test_date_read_as_a_number_is_kept(tmp_path, text, date, inside, entity_type):
    """
    Test that a numeric date another recognizer reads as a phone, license or account number is kept with the date.
    """
    scrubbed, report = scrub_with_report(text, [span(text, date, 'DATE_TIME'), span(text, inside, entity_type)],
                                         tmp_path)

    assert scrubbed == text
    assert report == {('DATE_TIME', date): False, (entity_type, inside): False}


@pytest.mark.parametrize('text, date, inside, entity_type', [
    ('Seen 3/14/23 call 541-555-0134 re labs.\n', '3/14/23 call 541-555-0134', '541-555-0134', 'PHONE_NUMBER'),
    ('Seen 3/14/23 1.541.555.0134\n', '3/14/23 1.541.555.0134', '1.541.555.0134', 'PHONE_NUMBER'),
    ('Seen 3/14/23 email jo@example.com\n', '3/14/23 email jo@example.com', 'jo@example.com', 'EMAIL_ADDRESS'),
    ("Bella's 5th birthday party.\n", "Bella's 5th birthday", 'Bella', 'PERSON'),
    # A subject ID the GRLS pattern missed, found by the model.
    ('Seen 094-0l2345 03/14/2023.\n', '094-0l2345 03/14/2023', '094-0l2345', 'ID'),
    ('Seen June 3 at 77 Old Mill Rd.\n', 'June 3 at 77 Old Mill Rd', '77 Old Mill Rd', 'ADDRESS'),
])
def test_identifying_text_inside_a_date_is_still_scrubbed(tmp_path, text, date, inside, entity_type):
    """
    Test that a phone number, email, name, ID or address the model took for part of a date is still scrubbed.
    """
    scrubbed, report = scrub_with_report(text, [span(text, date, 'DATE_TIME'), span(text, inside, entity_type)],
                                         tmp_path)

    assert inside not in scrubbed
    assert report[(entity_type, inside)] is True


def test_subject_id_inside_a_date_is_still_scrubbed(tmp_path):
    """
    Test that a GRLS subject ID is scrubbed even if the model reads it as part of a date.
    """
    text = 'Seen 094-012345 03/14/2023.\n'
    detections = [span(text, '094-012345 03/14/2023', 'DATE_TIME'),
                  span(text, '094-012345', 'ID', pii_scrubber.SUBJECT_ID_RECOGNIZER_NAME)]

    scrubbed, report = scrub_with_report(text, detections, tmp_path)

    assert '012345' not in scrubbed
    assert report[('DATE_TIME', '094-012345 03/14/2023')] is True


def test_date_overlapping_a_replaced_detection_is_reported_as_scrubbed(tmp_path):
    """
    Test that a kept date whose text was replaced (here as part of an address) is reported as scrubbed.
    """
    text = 'Clinic: 2015 Oak St, Bend, OR 97701\n'
    detections = [span(text, '2015 Oak St, Bend, OR 97701', 'ADDRESS'), span(text, '2015 Oak', 'DATE_TIME')]

    scrubbed, report = scrub_with_report(text, detections, tmp_path)

    assert scrubbed == 'Clinic: <ADDRESS>\n'
    assert report == {('ADDRESS', '2015 Oak St, Bend, OR 97701'): True, ('DATE_TIME', '2015 Oak'): True}


def test_kept_initials_are_listed_in_the_report(tmp_path):
    """
    Test that initials are left in the text but listed in the report as not scrubbed, so QA can check them.
    """
    text = '03/02/21 JO     Exam by Jennifer Ortiz.\n'
    detections = [span(text, 'JO', 'PERSON'), span(text, 'Jennifer Ortiz', 'PERSON')]

    scrubbed, report = scrub_with_report(text, detections, tmp_path)

    assert scrubbed == '03/02/21 JO     Exam by <PERSON>.\n'
    assert report == {('PERSON', 'JO'): False, ('PERSON', 'Jennifer Ortiz'): True}


def test_title_and_short_surname_is_scrubbed(analyzer):
    """
    Test that a title and a short surname in capitals is scrubbed, not kept as initials.
    """
    scrubbed, _ = pii_scrubber.scrub_pii('OWNER: MRS. COX CALLED RE: LABS.\n', analyzer, threshold=0.0)
    assert 'COX' not in scrubbed


@pytest.mark.parametrize('text', [
    'Rabies, SC\n',
    'Bordetella, IN\n',
    'Recheck 4 Weeks Dr Smith\n',
    'Referred for 1 Abdominal CT scan\n',
])
def test_medical_text_is_not_an_address(analyzer, text):
    """
    Test that vaccine routes, recheck intervals and imaging orders aren't scrubbed as addresses.
    """
    _, results = pii_scrubber.scrub_pii(text, analyzer, threshold=0.0)
    assert 'ADDRESS' not in [result.entity_type for result in results]


@pytest.mark.parametrize('street', ['1428 Larkspur Lane', '1428 Oak Dr'])
def test_address_split_between_pieces_is_still_scrubbed(analyzer, street):
    """
    Test that an address whose street and city lines fall in different pieces of a long text is still scrubbed.

    Only subject IDs are kept from being split between pieces, so the whole-address pattern can't match here: the
    street must be found at the end of one piece and the city, state and ZIP at the start of the next. "Dr" is only
    a street type at the end of a line, which here is also the end of its piece.
    """
    text = f'Clinic notes.\n{street}\nSpringfield, OR 97477\nWeight 31.2 kg\n'
    max_length = text.index('Springfield')
    # Check the pieces split where this test needs them to, so a change to split_text can't make it pass vacuously.
    assert [piece for _, piece in split_text(text, max_length, keep_together=pii_scrubber.SUBJECT_ID_RE)][:2] == \
        [f'Clinic notes.\n{street}\n', 'Springfield, OR 97477\n']

    scrubbed, results = pii_scrubber.scrub_pii(text, analyzer, threshold=0.0, max_length=max_length)

    for part in street.split() + ['Springfield', 'OR 97477']:
        assert part not in scrubbed
    assert 'Clinic notes.' in scrubbed and 'Weight 31.2 kg' in scrubbed
    assert scrubbed.count('\n') == text.count('\n')
    assert [result.entity_type for result in results].count('ADDRESS') == 2


@pytest.mark.parametrize('entity_type, part, expected', [
    # A date's words or numbers read as something else: kept with the date.
    ('PERSON', 'June', True),
    ('ID', '2023', True),
    ('LOCATION', 'June 3', True),
    ('PHONE_NUMBER', '06-03-23', True),
    ('US_DRIVER_LICENSE', '060323', True),
    # Not only a date's words and numbers.
    ('PERSON', 'Seen', False),
    ('PHONE_NUMBER', '541-555-0134', False),
    ('US_BANK_NUMBER', '331423', False),
    # Only partly inside the date, or outside it.
    ('PERSON', 'June 3, 2023 by', False),
    ('PERSON', 'Ortiz', False),
])
def test_is_inside_kept_date(entity_type, part, expected):
    """
    Test which detections are kept as part of a date.
    """
    text = 'Seen June 3, 2023 by Ortiz 06-03-23 060323 541-555-0134 331423'
    kept_dates = pii_scrubber.SpanIndex([RecognizerResult('DATE_TIME', text.index('June'), text.index(' by'), 1.0),
                                         RecognizerResult('DATE_TIME', text.index('06-'), len(text), 1.0)])
    start = text.index(part)
    result = RecognizerResult(entity_type, start, start + len(part), 0.9)

    assert pii_scrubber.is_inside_kept_date(result, text, kept_dates) is expected
