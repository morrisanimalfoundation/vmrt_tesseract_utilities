"""
Tests for `pii_patterns`, the address and URL patterns and the initials check the PII scrubber uses.

These only need the standard library, so they run everywhere. `pii_scrubber_test.py` checks the same cases through
the real scrubber in the workspace image. All of the data below is made up.
"""

import pytest

from vmrt_tesseract_utilities.pii_patterns import (address_around,
                                                   find_addresses, find_urls,
                                                   is_date_part, is_initials)


@pytest.mark.parametrize('text, expected', [
    # Whole addresses, on one line or split after the street.
    ('1428 Larkspur Lane, Springfield, OR 97477', ['1428 Larkspur Lane, Springfield, OR 97477']),
    ('Clinic: Cedar Ridge Pet Hospital, 77 Old Mill Rd, Bend OR 97701', ['77 Old Mill Rd, Bend OR 97701']),
    ('220 W. Harbor Blvd Suite 4, Ventura, California 93001',
     ['220 W. Harbor Blvd Suite 4, Ventura, California 93001']),
    ('1428 Larkspur Lane\nSpringfield, OR 97477-1234', ['1428 Larkspur Lane\nSpringfield, OR 97477-1234']),
    ('PO Box 112, Bend, OR 97709', ['PO Box 112, Bend, OR 97709']),
    ('1428 LARKSPUR LN\nSPRINGFIELD OR 97477', ['1428 LARKSPUR LN\nSPRINGFIELD OR 97477']),
    ('61450 Hwy 97, Bend, OR 97702', ['61450 Hwy 97, Bend, OR 97702']),
    ('500 Main St NE #200, Salem, Oregon 97301', ['500 Main St NE #200, Salem, Oregon 97301']),
    # "Dr" and "Ct" as street types, where the address clearly goes on or ends.
    ('Clinic at 1428 Oak Dr, Bend, OR 97701', ['1428 Oak Dr, Bend, OR 97701']),
    ('1428 Oak Dr Bend OR 97701', ['1428 Oak Dr Bend OR 97701']),
    ('1428 OAK DR.\nBEND, OR 97701', ['1428 OAK DR.\nBEND, OR 97701']),
    ('22 Willow Ct Suite 3', ['22 Willow Ct Suite 3']),
    ('22 Willow Ct NW, Salem', ['22 Willow Ct NW']),
    ('Moved to 22 Willow Ct\n', ['22 Willow Ct']),
    ('22 WILLOW CT\n', ['22 WILLOW CT']),
    ('1428 Oak Dr N\n', ['1428 Oak Dr N']),
    ('1428 Maple Grove Ct (541) 555-0134', ['1428 Maple Grove Ct']),  # A letterhead's phone number after it.
    # Other abbreviated street types, where the address goes on or ends.
    ('7 Old Mill Rd, Bend OR 97701', ['7 Old Mill Rd, Bend OR 97701']),
    # A city, state and ZIP a few words further on the same line.
    ('12345 N. Elm St. Two Rivers Bend, OR 97701', ['12345 N. Elm St. Two Rivers Bend, OR 97701']),
    ('5 Elm St Cedar Ridge Clinic Bend OR 97701', ['5 Elm St', 'Ridge Clinic Bend OR 97701']),
    ('5 Elm St\nBend, OR 97701', ['5 Elm St\nBend, OR 97701']),
    ('Moved to 14 Elm St.\n', ['14 Elm St.']),
    ('14 Larkspur Ln, Bend', ['14 Larkspur Ln']),
    ('1428 LARKSPUR LN\n', ['1428 LARKSPUR LN']),
    ('9 Town Sq Suite 2', ['9 Town Sq Suite 2']),
    ('22 Willow Pl NE', ['22 Willow Pl NE']),
    # Written-out street types, and abbreviations that aren't medical ones, count anywhere, even after one digit.
    ('5 Elm Street near the park', ['5 Elm Street']),
    ('3 Kings Way, Bend', ['3 Kings Way']),
    ('12345 W. Juniper Rd. Two Rivers', ['12345 W. Juniper Rd.']),
    ('Home: 742 OAK RD BEND', ['742 OAK RD']),
    ('1428 Larkspur Ave N Bend', ['1428 Larkspur Ave N']),
    ('Clinic at 220 N.E. 3rd Ave. Bend', ['220 N.E. 3rd Ave.']),
    ('220 1st 3rd Ave apt', ['220 1st 3rd Ave']),
    ('Mail to 99 Harbor Blvd today', ['99 Harbor Blvd']),
    # Parts of an address on their own.
    ('Located at 1428 Larkspur Lane near the park', ['1428 Larkspur Lane']),
    ('Springfield, OR 97477', ['Springfield, OR 97477']),
    ('OR 97477', ['OR 97477']),
    ('Boise, ID 83702', ['Boise, ID 83702']),  # "ID" is Idaho after a city and a comma.
    ('1428 Oak St, Boise, ID 83702', ['1428 Oak St, Boise, ID 83702']),
    ('Bend, Oregon\n', ['Bend, Oregon']),
    ('Salem, NEW YORK;', ['Salem, NEW YORK']),
    ('Referred to a clinic in Bend, Oregon.', []),  # Left to the model: the full stop could end a sentence.
])
def test_addresses_are_found(text, expected):
    """
    Test that whole addresses are matched as one, and parts of one on their own.
    """
    assert find_addresses(text) == expected


@pytest.mark.parametrize('text', [
    'Recheck in 6 months OR sooner if needed.',
    'Taken to OR at 10:30 for the spay.',
    'Weight 31.2 kg. Temperature 101.4 F.',
    'Give 2 tabs by mouth twice daily.',
    'Seen 2 times by Dr. Whitaker.',
    'Rabies 3 Yr given, lot 12345.',
    'Mild dental tartar noted on upper premolars.',
    'Pt is lethargic, or not eating.',
    'WBC 12.5, PCV 45, TP 7.2',
    # State abbreviations after a comma with no ZIP code: vaccine routes and exam notes, not states.
    'Rabies, SC',
    'Bordetella, IN\n',
    'Attitude Good, OK',
    'Bend, OR',
    # "Dr" and "Ct" followed by more words: a doctor's name and a CT scan, not streets.
    'Recheck 4 Weeks Dr Smith',
    'RECHECK 2 WEEKS DR SMITH',
    'Recheck 4 Weeks Dr. Smith',
    'Referred for 1 Abdominal CT scan',
    'REFERRED FOR 1 ABDOMINAL CT SCAN',
    # A doctor's initial or "W/" for "with" isn't a direction after "Dr" or "CT".
    'Recheck 2 Wks Dr N. Patel',
    '1 ABDOMINAL CT W/CONTRAST',
    '1 Thoracic CT W/ contrast',
    # "Dr" and "CT" ending a line or clause: a time or a scan, or in capitals after an ordinary word.
    'Performed 1 Abdominal CT',
    'PERFORMED 1 ABDOMINAL CT',
    '2 Thoracic CT, 1 Abdominal CT scan',
    'Recheck 2 Weeks Dr, then',
    'Recheck 2 WEEKS DR\n',
    'Ordered 1 Skull CT\n',
    # Medical abbreviations that are also abbreviated street types: subcutaneous, lymph node, ST segment.
    'DAPP 1 ML SQ given',
    'Fluids: 300 ML LRS SQ',
    'Rabies 1 Yr Vaccine SQ RH',
    '1 Enlarged Popliteal LN',
    '1 ENLARGED POPLITEAL LN\n',
    'Mandibular 2 Cm LN noted',
    '2 Mm ST Elevation',
    'ECG: 2 MM ST ELEVATION',
    # A dose, size or time after a number isn't a house number, even before a written-out street type.
    'Rx: 1 Tab PO Way',
    'Give 2 Tabs With Food Daily, Drive slowly',
    # "ID" is an identifier's label, not Idaho.
    'Microchip ID 98102 implanted',
    'Patient ID 12345',
    'ID 83702',
])
def test_ordinary_text_is_not_an_address(text):
    """
    Test that ordinary medical text, including state abbreviations that are also medical ones, isn't taken for an
    address.
    """
    assert find_addresses(text) == []


@pytest.mark.parametrize('text, found, expected', [
    # The rest of a street after an address that stopped short.
    ('PO Box 112 Two Rivers Rd\nBend', 'PO Box 112', 'PO Box 112 Two Rivers Rd'),
    ('1428 Larkspur Lane Two Rivers Rd', '1428 Larkspur Lane', '1428 Larkspur Lane Two Rivers Rd'),
    # A house number, direction and suite around a street the model only found the name of.
    ('1234 N Cedar Ridge STE 1000', 'Cedar Ridge', '1234 N Cedar Ridge STE 1000'),
    ('1234 N Cedar Ridge E STE 100A', 'Cedar Ridge', '1234 N Cedar Ridge E STE 100A'),
    ('1234W. Broadway., Ste. 12', 'Broadway', '1234W. Broadway., Ste. 12'),
    ('1234 N Cedar Ridge STE 1000', 'Cedar Ridge STE', '1234 N Cedar Ridge STE 1000'),  # Found up to the unit.
    ('Bend PASTE 1000', 'Bend PASTE', 'Bend PASTE'),
    ('Clinic: 500 Main St; Suite 12', '500 Main St', '500 Main St; Suite 12'),
    # A suite after a word, such as a building's name.
    ('500 Main St Medical Ste 100A', '500 Main St', '500 Main St Medical Ste 100A'),
    ('500 Main St    Plaza Suite 4', '500 Main St', '500 Main St    Plaza Suite 4'),
    # Not a bare "#" after a word: that's another kind of number.
    ('500 Main St Invoice #12345', '500 Main St', '500 Main St'),
    ('500 Main St Unit price 12', '500 Main St', '500 Main St'),
    # A ZIP code after a city, with or without the state.
    ('Ann Lee, Springfield 97477 DL 1234', 'Springfield', 'Springfield 97477'),
    ('Ann Lee, Springfield OR. 97477', 'Springfield', 'Springfield OR. 97477'),
    # A street's name the model found the middle of.
    ('Maple Grove Ct (541) 555-0134', 'Grove', 'Maple Grove Ct'),
    ('Address: Grove Ct 541-555-0134', 'Grove', 'Grove Ct'),
    # Nothing next to it is part of an address.
    ('Bend\n97701', 'Bend', 'Bend'),
    ('Referred to Bend CT scan', 'Bend', 'Bend'),
    ('Recheck in Bend Dr Smith', 'Bend', 'Bend'),
    ('3/14/2024 Bend', 'Bend', 'Bend'),
    ('Bend 12345A lot', 'Bend', 'Bend'),
    ('Ph 5415550134 Bend', 'Bend', 'Bend'),
    ('Moved to Bend in 2024', 'Bend', 'Bend'),
    ('Heartgard 6 ct Bend', 'Bend', 'Bend'),
    ('Referred to Bend CT W/CONTRAST', 'Bend', 'Bend'),
])
def test_address_around(text, found, expected):
    """
    Test that a found address or place is stretched over the address parts right next to it, and only those.
    """
    start = text.index(found)
    new_start, new_end = address_around(text, start, start + len(found))
    assert text[new_start:new_end] == expected


def test_street_name_is_not_taken_before_a_whole_address():
    """
    Test that a whole street address isn't stretched back over the words before it, such as a clinic's name.
    """
    text = 'Cedar Ridge Pet Hospital 77 Old Mill Rd Two Rivers Rd'
    start = text.index('77')
    new_start, new_end = address_around(text, start, text.index(' Two'), name_may_start_before=False)
    assert text[new_start:new_end] == '77 Old Mill Rd Two Rivers Rd'


@pytest.mark.parametrize('text, expected', [
    ('Rads: https://pacs.sunnymeadowsvet.com/study/8812 reviewed', ['https://pacs.sunnymeadowsvet.com/study/8812']),
    ('See http://192.168.1.20/xray/44', ['http://192.168.1.20/xray/44']),
    ('Images at www.vetimaging.net/x/1', ['www.vetimaging.net/x/1']),
    ('Portal: sunnymeadowsvet.com/xray', ['sunnymeadowsvet.com/xray']),
    ('Visit SUNNYMEADOWS.COM for hours', ['SUNNYMEADOWS.COM']),
    ('Results posted to myclinic.org.', ['myclinic.org']),
    # TLDs that are also words count with a path after them.
    ('sunnymeadows.pet/records', ['sunnymeadows.pet/records']),
    ('Visit bendanimalclinic.us/hours', ['bendanimalclinic.us/hours']),
    ('SUNNYMEADOWS.VET/CONTACT', ['SUNNYMEADOWS.VET/CONTACT']),
    ('Book at bayview.clinic/book today', ['bayview.clinic/book']),
    # Any TLD counts with a scheme or www.
    ('Report at https://lab.example.do/r/1', ['https://lab.example.do/r/1']),
    ('www.clinic.today', ['www.clinic.today']),
    # A comma typed for a dot, after www.
    ('Visit www.sunnymeadows,vet for hours', ['www.sunnymeadows,vet']),
    ('Visit www,sunnymeadows.com', ['www,sunnymeadows.com']),
])
def test_urls_are_found(text, expected):
    """
    Test that URLs with a scheme, www. or a common TLD are matched.
    """
    assert find_urls(text) == expected


@pytest.mark.parametrize('text', [
    'Pt is lethargic.do not vaccinate today.',
    'Owner reports dog is quiet.today.',
    'Owner called. vomiting.no diarrhea',
    'Recheck in 2 wks.Net weight 31 kg',
    'Supportive care.Care plan reviewed.',
    'Recheck if worse.Us to call owner.',
    'Discussed with Dr.Smith at length.',
    'Dx: otitis externa.Tx: ear cleaning.',
    # Words joined to a TLD that is also a word, with no path after it.
    'PT VOMITING.DOG WAS QUIET',
    'MOUTH.US was',
    'HOME.CARE PLAN',
    'seen by vet.dog ok',
    'pet health.care plan',
    'given.co 2',
    'Book at bayview.clinic',
    'SUNNYMEADOWS.VET',
    'Email margaret.thompson@example.com',  # Left to the email recognizer.
])
def test_ordinary_text_is_not_a_url(text):
    """
    Test that words joined by a full stop, as OCR leaves them when it loses a space, aren't taken for URLs.
    """
    assert find_urls(text) == []


@pytest.mark.parametrize('name', ['JO', 'AW', 'J.O.', 'J. M. O.', 'K'])
def test_initials_are_recognized(name):
    """
    Test that initials, such as a staff column's, are recognized so they're kept.
    """
    assert is_initials(name)


@pytest.mark.parametrize('name', ['Jo', 'Ortiz', 'Jennifer Ortiz', 'Dr. Whitaker', 'JENNIFER', 'J Ortiz', 'J. Ortiz',
                                  # Three capitals could be a name in an all-caps record.
                                  'TOM', 'KIM', 'JMO',
                                  # After a title, short capitals are a surname.
                                  'MRS. COX', 'MR. LEE', 'DR. NG', 'Dr. AW', 'Ms JO'])
def test_names_are_not_initials(name):
    """
    Test that a first or last name, even a short one, isn't taken for initials, so it's scrubbed.
    """
    assert not is_initials(name)


@pytest.mark.parametrize('text', ['June', 'June 3', 'JUNE', 'Tues.', '3rd', '2023', '3/14/2023', '03-14-2023',
                                  'Mar. 14, 2023', 'Monday, June 3', '03-14-23', '031423', '03142023', '20230314'])
def test_date_parts_are_recognized(text):
    """
    Test that a date's words and numbers, which are sometimes read as a name, ID or phone number, are recognized so
    they're kept with the date.
    """
    assert is_date_part(text)


@pytest.mark.parametrize('text', ['Jennifer', 'June Ortiz', '094-012345', '541-555-0134', '555-0134', '541',
                                  '12345', 'Bella', 'jo@example.com', '1428 Oak St', '1.541.555.0134',
                                  # Six or eight digits that aren't a valid month and day.
                                  '331423', '13142023', '123456789'])
def test_other_text_is_not_a_date_part(text):
    """
    Test that a name, subject ID, phone number or address inside a date isn't taken for part of it, so it's still
    scrubbed.
    """
    assert not is_date_part(text)
