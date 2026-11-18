"""
Patterns the PII scrubber uses for addresses and URLs, and the check for a name that is only initials.

These are plain regexes, so they can be tested without Presidio, which is only installed in the workspace image.
The scrubber wraps them in Presidio pattern recognizers.
"""

import re

# US states, DC and the territories, as USPS abbreviations and full names.
US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado',
    'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan',
    'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin',
    'WY': 'Wyoming', 'PR': 'Puerto Rico', 'GU': 'Guam', 'VI': 'Virgin Islands',
}

# The patterns below are case-sensitive unless a part says otherwise with (?i:...), because a state abbreviation
# only counts in capitals: "OR" is Oregon, "or" is a word. Compile them with re.MULTILINE and without re.IGNORECASE.
ADDRESS_REGEX_FLAGS = re.MULTILINE

# A state's name, in any case, such as "Oregon" or "NEW YORK".
_STATE_NAME = r'(?i:' + '|'.join(name.replace(' ', r'[ \t]+') for name in US_STATES.values()) + r')\b'

# A state: its abbreviation in capitals (with or without a full stop), or its name.
_STATE = r'(?:(?:' + '|'.join(US_STATES) + r')\b\.?|' + _STATE_NAME + ')'

_ZIP = r'[0-9]{5}(?:-[0-9]{4})?(?![0-9])'

# One to three capitalized words, such as "Springfield", "North Bend" or "ST. LOUIS".
_CITY = r"[A-Z][A-Za-z.'-]*(?:[ \t]+[A-Z][A-Za-z.'-]*){0,2}"

# A state, but not "ID" for Idaho, which is more often an identifier's label ("Patient ID 12345").
_STATE_NOT_ID = r'(?:(?:' + '|'.join(state for state in US_STATES if state != 'ID') + r')\b\.?|' + _STATE_NAME + ')'

# "Springfield, OR 97477", "Bend OR 97701", "OR 97701" or "Ventura, California 93001". "ID" only counts after a city
# and a comma ("Boise, ID 83702"), so "Microchip ID 98102" isn't an address.
CITY_STATE_ZIP_REGEX = (rf'(?<![A-Za-z])(?:(?:{_CITY},?[ \t]+)?{_STATE_NOT_ID}|{_CITY},[ \t]*ID\b\.?)'
                        rf',?[ \t]+{_ZIP}')

# "Bend, Oregon" with nothing after it on the line but punctuation. Without a ZIP code, only a state's full name
# counts: abbreviations after a comma are too often something else ("Rabies, SC", "Bordetella, IN", "Attitude, OK").
CITY_STATE_REGEX = rf'(?<![A-Za-z]){_CITY},[ \t]*{_STATE_NAME}(?=[ \t]*(?:[,;)]|$))'

# A house number that isn't part of a longer number, a decimal, a date or a measurement.
_HOUSE_NUMBER = r'(?<![\w.,/:-])[0-9]{1,6}[A-Z]?(?:-[0-9]{1,4})?'

# A word of a street's name: capitalized or a number, such as "Larkspur", "W." or "5th".
_STREET_WORD = r"[A-Z0-9][A-Za-z0-9'.-]*"

_UNIT_WORD = r'(?:(?=[A-Z])(?i:suite|ste|unit|apt|bldg|building)\b\.?|#)'

_UNIT = rf'(?:,?[ \t]*{_UNIT_WORD}[ \t]*#?[A-Z0-9-]+\b)?'

_DIRECTION = r'[NSEW]{1,2}\b\.?'

# Street types written out, in any case but starting with a capital, so "lane" in a sentence isn't one.
_STREET_TYPE_WORDS = ('street', 'avenue', 'road', 'boulevard', 'lane', 'drive', 'court', 'circle', 'way', 'place',
                      'parkway', 'terrace', 'trail', 'loop', 'pike', 'square', 'plaza')
_STREET_TYPE = r'(?=[A-Z])(?i:' + '|'.join(_STREET_TYPE_WORDS) + r')\b\.?'

# Abbreviated street types. Many are also medical abbreviations or something else ("Dr Smith", "1 Abdominal CT",
# "1 ML SQ", "Popliteal LN", "2 Mm ST Elevation"), so they only count where the address clearly goes on after them,
# or where they end a line or clause (see _street_name_then_type).
_STREET_TYPE_ABBREVIATIONS = ('st', 'ave', 'rd', 'blvd', 'ln', 'dr', 'ct', 'cir', 'pl', 'pkwy', 'ter', 'trl', 'sq',
                              'plz')
_ABBREVIATED_STREET_TYPE = r'(?=[A-Z])(?i:' + '|'.join(_STREET_TYPE_ABBREVIATIONS) + r')\b\.?'
_TITLE_CASE_STREET_TYPE = '(?:' + '|'.join(a.capitalize() for a in _STREET_TYPE_ABBREVIATIONS) + r')\b\.?'
_CAPS_STREET_TYPE = '(?:' + '|'.join(a.upper() for a in _STREET_TYPE_ABBREVIATIONS) + r')\b\.?'

# A US phone number, such as "(541) 555-0134" or "541.555.0134".
_PHONE = r'\(?[0-9]{3}\)?[ \t.-]?[0-9]{3}[ \t.-]?[0-9]{4}(?![0-9])'

# Where an address clearly goes on after an abbreviated street type: a direction that ends the street, a unit, the
# city, state and ZIP (after a space, comma or line break), or a phone number, as in a letterhead. A direction must
# end the street, so a doctor's initial ("Dr N. Patel") or "W/" for "with" ("CT W/CONTRAST") isn't one.
_LINE_OR_CLAUSE_END = r'(?=[ \t]*(?:[,;]|$))'
_BEFORE_REST = r'(?:[ \t]*,[ \t]*|[ \t]*\r?\n[ \t]*|[ \t]+)'
_CLEAR_ADDRESS_AFTER = (rf'(?=[ \t]+{_DIRECTION}(?:{_LINE_OR_CLAUSE_END}|[ \t]+(?:{_UNIT_WORD}|{CITY_STATE_ZIP_REGEX}))'
                        rf'|{_BEFORE_REST}(?:{_UNIT_WORD}|{CITY_STATE_ZIP_REGEX})|[ \t]+{_PHONE})')

# A word that is never part of a street's name before an abbreviated street type: a time ("Recheck 2 Weeks Dr") or
# what a CT scan is of ("1 Abdominal CT").
_NOT_STREET_NAME = (r'(?i:days?|wks?|weeks?|mos?|months?|yrs?|years?|hrs?|hours?|abdominal|abdomen|thoracic|thorax|'
                    r'chest|head|skull|brain|spinal|spine|pelvic|pelvis|nasal|sinus|contrast)\b')
_AMBIGUOUS_NAME_WORD = rf'(?!{_NOT_STREET_NAME}){_STREET_WORD}'
_CAPS_NAME_WORD = rf"(?!{_NOT_STREET_NAME})[A-Z0-9][A-Z0-9'.-]*(?![a-z])"

# A unit of measure or time right after a number, which makes it a dose, size or time rather than a house number
# ("1 ML SQ", "2 Cm LN", "1 Yr Vaccine SQ", "1 Tab PO Way").
_MEASURE = (r'(?i:mls?|mg|mcg|ug|cc|cm|mm|kgs?|lbs?|gm|oz|tabs?|tablets?|caps?|capsules?|units?|iu|drops?|gtts?|'
            r'doses?|days?|wks?|weeks?|mos?|months?|yrs?|years?|hrs?|hours?|mins?|minutes?)\b')


def _street_name_then_type(name_of, at_line_end: bool = True) -> str:
    """
    The regex for a street's name followed by its type, where name_of(word) is the regex for the name made of words.

    A written-out type counts anywhere. An abbreviated one counts where the address clearly goes on after it, and,
    if at_line_end, also at the end of a line or before a comma or semicolon, but only written in the same case as
    the name ("22 Willow Ct", "22 WILLOW CT"), since a capitalized abbreviation after an ordinary word ("Performed 1
    Abdominal CT", "1 Enlarged Popliteal LN") is a medical one.
    """
    line_end = (rf'|{name_of(_AMBIGUOUS_NAME_WORD)}{_TITLE_CASE_STREET_TYPE}{_LINE_OR_CLAUSE_END}'
                rf'|{name_of(_CAPS_NAME_WORD)}{_CAPS_STREET_TYPE}{_LINE_OR_CLAUSE_END}') if at_line_end else ''
    return (rf'(?:{name_of(_STREET_WORD)}{_STREET_TYPE}'
            rf'|{name_of(_AMBIGUOUS_NAME_WORD)}{_ABBREVIATED_STREET_TYPE}{_CLEAR_ADDRESS_AFTER}'
            rf'{line_end})')


# Roads named by number, such as "Hwy 97" or "Route 9A".
_NUMBERED_ROAD = r'(?=[A-Z])(?i:highway|hwy|route|rte)\b\.?[ \t]+[0-9]{1,4}[A-Z]?\b'

# "1428 Larkspur Lane", "220 W. Harbor Blvd Suite 4", "500 Main St NE" or "61450 Hwy 97". A street type needs at
# least one word of name before it, so "2 Dr. Smith" isn't an address. An abbreviated street type ending a line
# needs a house number of two or more digits, since a single digit there is more often a count ("1 ENLARGED
# POPLITEAL LN").
_STREET_NAME_AND_TYPE = _street_name_then_type(lambda word: rf'(?:{word}[ \t]+){{1,4}}', at_line_end=False)
_STREET_NAME_AND_TYPE_AT_LINE_END = _street_name_then_type(lambda word: rf'(?:{word}[ \t]+){{1,4}}')
STREET_ADDRESS_REGEX = (rf'(?:{_HOUSE_NUMBER}[ \t]+(?!{_MEASURE})'
                        rf'(?:{_STREET_NAME_AND_TYPE}(?:[ \t]+{_DIRECTION})?'
                        rf'|(?:{_STREET_WORD}[ \t]+){{0,3}}{_NUMBERED_ROAD})'
                        rf'|(?=[0-9]{{2}}){_HOUSE_NUMBER}[ \t]+(?!{_MEASURE}){_STREET_NAME_AND_TYPE_AT_LINE_END})'
                        rf'{_UNIT}')

PO_BOX_REGEX = r'\b(?i:p\.?[ \t]*o\.?[ \t]*box)[ \t]+[0-9]+\b'

# A street or PO box followed by its city, state and ZIP, on the same line or the next.
FULL_ADDRESS_REGEX = (rf'(?:{STREET_ADDRESS_REGEX}|{PO_BOX_REGEX})(?:[ \t]*,[ \t]*|[ \t]*\r?\n[ \t]*|[ \t]+)'
                      rf'{CITY_STATE_ZIP_REGEX}')

# (name, regex, score) for each address pattern. A whole address scores higher than its parts, so it's preferred.
ADDRESS_PATTERNS = [
    ('US address', FULL_ADDRESS_REGEX, 0.95),
    ('US street address', STREET_ADDRESS_REGEX, 0.9),
    ('PO box', PO_BOX_REGEX, 0.9),
    ('US city, state and ZIP', CITY_STATE_ZIP_REGEX, 0.9),
    ('US city and state', CITY_STATE_REGEX, 0.85),
]

# The parts of an address that can be left next to one the model or the patterns above found only part of: the rest
# of its street, a unit and a ZIP code after it, and a house number and direction before it. See address_around.
_STREET_REST_NAME_AND_TYPE = _street_name_then_type(lambda word: rf'(?:[ \t]+{word}){{0,3}}[ \t]+')
_STREET_REST = rf'{_STREET_REST_NAME_AND_TYPE}(?:[ \t]+{_DIRECTION})?'
_UNIT_AFTER = rf'\.?[ \t]*[,;]?[ \t]*(?:{_DIRECTION}[ \t]+)?{_UNIT_WORD}[ \t]*#?[A-Z0-9-]+\b'
_ZIP_AFTER = rf'\.?(?:,?[ \t]+{_STATE})?,?[ \t]+{_ZIP}(?![\w-])'
_AFTER_ADDRESS_RE = re.compile(rf'(?P<street>{_STREET_REST})?(?:{_UNIT_AFTER})?(?:{_ZIP_AFTER})?',
                               ADDRESS_REGEX_FLAGS)
# A span that ends with a unit word, but not its number ("Juniper Ridge STE" in "Juniper Ridge STE 1000").
_UNIT_WORD_AT_END_RE = re.compile(rf'(?<![A-Za-z]){_UNIT_WORD}\Z', ADDRESS_REGEX_FLAGS)
_UNIT_NUMBER_RE = re.compile(r'[ \t]*#?[A-Z0-9-]+\b', ADDRESS_REGEX_FLAGS)

_NUMBER_BEFORE = r'(?<![\w.,/:-])[0-9]{1,6}(?:[A-Z]\.?|-[0-9]{1,4})?[ \t]+'
_HOUSE_NUMBER_BEFORE_RE = re.compile(rf'{_NUMBER_BEFORE}(?:{_DIRECTION}[ \t]+)?\Z', ADDRESS_REGEX_FLAGS)
# With the rest of a street after it, a place can also be the middle of a street's name ("Maple <Grove> Ct").
_STREET_BEFORE_RE = re.compile(rf'(?:{_NUMBER_BEFORE})?(?:{_DIRECTION}[ \t]+)?'
                               rf"(?:(?<![\w'.-]){_STREET_WORD}[ \t]+){{1,2}}\Z", ADDRESS_REGEX_FLAGS)

# The URL patterns are case-insensitive except where a part says otherwise with (?-i:...).
URL_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE

_URL_TAIL = r'''(?:[/?#][^\s<>"']*)?'''
_HOST_LABEL = r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?'

# Top-level domains a bare domain (with no scheme or www.) must end in to count as a URL. Many short words are also
# TLDs (.do, .no, .today), so OCR'd text that loses the space after a full stop ("lethargic.do not") would otherwise
# read as a URL. The TLD must be in lowercase or capitals, since one in title case ("wks.Net weight") is more likely
# a missing space before a new sentence.
BARE_DOMAIN_TLDS = ('com', 'net', 'org', 'edu', 'gov', 'info', 'biz')

# Top-level domains that are also ordinary words in a vet record ("VOMITING.DOG", "HOME.CARE", "MOUTH.US"). A bare
# domain ending in one only counts as a URL with a path after it ("cedarridge.vet/contact").
WORD_TLDS = ('us', 'co', 'vet', 'pet', 'dog', 'clinic', 'hospital', 'health', 'care')


def _tld(tlds: tuple[str, ...]) -> str:
    return '(?-i:' + '|'.join(tlds + tuple(tld.upper() for tld in tlds)) + ')'


_BARE_DOMAIN = rf'(?<![\w.@-]){_HOST_LABEL}(?:\.{_HOST_LABEL})*\.'

# (name, regex, score) for each URL pattern.
URL_PATTERNS = [
    ('URL with scheme', r'''\b(?:https?|ftp)://[^\s<>"']+''', 0.6),
    ('www URL', rf'\bwww[0-9]{{0,3}}\.{_HOST_LABEL}(?:\.{_HOST_LABEL})+{_URL_TAIL}', 0.6),
    ('Bare domain', rf'{_BARE_DOMAIN}{_tld(BARE_DOMAIN_TLDS)}\b(?![.-]?\w){_URL_TAIL}', 0.5),
    ('Bare domain with a path', rf'''{_BARE_DOMAIN}{_tld(WORD_TLDS)}/[^\s<>"']*''', 0.5),
]

# A name that is only initials: one or two capitals ("K", "JO") or up to three letters each with a full stop ("J.O.",
# "J. M. O."). Three capitals in a row could be a name in an all-caps record ("TOM", "KIM"), and anything after a
# title ("MRS. COX", "Dr. AW") is taken for a surname, so neither counts.
_INITIALS_RE = re.compile(r'[A-Z]{1,2}|(?:[A-Z]\.[ \t]?){1,3}')

# Words and numbers a date is written with: month and weekday names, a day or month of one or two digits (with or
# without "st", "nd" and so on), a four-digit year, and a whole date written as one number ("031423", "03142023",
# "20230314") if it's a valid month and day. A three-digit number never is, so part of a phone number ("555-0134")
# isn't one.
_MONTH_OR_DAY_NAME = (r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|'
                      r'sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|'
                      r'mon(?:day)?|tues?(?:day)?|wed(?:nesday)?|thu(?:rs?)?(?:day)?|fri(?:day)?|sat(?:urday)?|'
                      r'sun(?:day)?)\b\.?')
_MONTH_DAY = r'(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])'
_COMPACT_DATE = rf'(?:{_MONTH_DAY}(?:[0-9]{{2}}|(?:19|20)[0-9]{{2}})|(?:19|20)[0-9]{{2}}{_MONTH_DAY})\b'
_DATE_WORD = rf'(?:{_MONTH_OR_DAY_NAME}|{_COMPACT_DATE}|(?:[0-9]{{1,2}}|[0-9]{{4}})(?:st|nd|rd|th)?\b)'
_DATE_PART_RE = re.compile(rf'{_DATE_WORD}(?:[ \t,./-]+{_DATE_WORD})*', re.IGNORECASE)


def find_addresses(text: str) -> list[str]:
    """
    Finds the addresses in text, joining overlapping matches, so a whole address is found as one.

    This is for testing the patterns. The scrubber uses Presidio, which resolves overlapping matches its own way.

    Parameters
    ----------
    text : str
        The text to search.

    Returns
    -------
    list[str]
        The matched text of each address, in order.

    Examples
    --------
    >>> find_addresses('1428 Larkspur Lane, Springfield, OR 97477')
    ['1428 Larkspur Lane, Springfield, OR 97477']
    >>> find_addresses('Recheck in 6 months OR sooner if needed.')
    []
    """
    return _find_longest(text, [regex for _, regex, _ in ADDRESS_PATTERNS], ADDRESS_REGEX_FLAGS)


def address_around(text: str, start: int, end: int, name_may_start_before: bool = True) -> tuple[int, int]:
    """
    Stretches the span of an address, or a place that could be part of one, over the address parts right next to it
    on the same line: the rest of its street, a unit and a ZIP code after it, and a house number and direction
    before it. The model often finds only the street's or city's name, which would leave the rest of the address in.

    Parameters
    ----------
    text : str
        The whole text.
    start, end : int
        The span found.
    name_may_start_before : bool
        Whether the street's name may start before the span, when the rest of a street follows it ("Maple <Grove>
        Ct"). Not for a span that is already a whole street address.

    Returns
    -------
    tuple[int, int]
        The stretched span, the same as the one given if nothing next to it is part of an address.

    Examples
    --------
    >>> text = '1234 N Cedar Ridge STE 100, Bend 97701'
    >>> start, end = address_around(text, 7, 18)  # "Cedar Ridge"
    >>> text[start:end]
    '1234 N Cedar Ridge STE 100'
    >>> start, end = address_around(text, 28, 32)  # "Bend"
    >>> text[start:end]
    'Bend 97701'
    >>> address_around('Referred to Bend CT scan', 12, 16)
    (12, 16)
    """
    if _UNIT_WORD_AT_END_RE.search(text, start, end):
        number = _UNIT_NUMBER_RE.match(text, end)
        end = number.end() if number else end
    after = _AFTER_ADDRESS_RE.match(text, end)
    window = max(0, start - 80)
    before = None
    if after.group('street') and name_may_start_before:
        before = _STREET_BEFORE_RE.search(text, window, start)
    if before is None:
        before = _HOUSE_NUMBER_BEFORE_RE.search(text, window, start)
    return (before.start() if before else start), after.end()


def find_urls(text: str) -> list[str]:
    """
    Finds the URLs in text, joining overlapping matches.

    Parameters
    ----------
    text : str
        The text to search.

    Returns
    -------
    list[str]
        The matched text of each URL, in order.

    Examples
    --------
    >>> find_urls('Rads at https://pacs.example.com/study/1 today.')
    ['https://pacs.example.com/study/1']
    >>> find_urls('Pt is lethargic.do not vaccinate.')
    []
    """
    return _find_longest(text, [regex for _, regex, _ in URL_PATTERNS], URL_REGEX_FLAGS)


def _find_longest(text: str, regexes: list[str], flags: int) -> list[str]:
    """
    Finds every match of the regexes and joins any that overlap into one span.

    This is simpler than how Presidio resolves overlapping matches, but enough to test what the patterns match
    together.

    Parameters
    ----------
    text : str
        The text to search.
    regexes : list[str]
        The regexes.
    flags : int
        The flags to compile them with.

    Returns
    -------
    list[str]
        The matched text, in order.
    """
    spans = sorted({match.span() for regex in regexes for match in re.finditer(regex, text, flags)
                    if match.end() > match.start()}, key=lambda span: (span[0], -span[1]))
    kept = []
    for start, end in spans:
        if kept and start < kept[-1][1]:
            kept[-1] = (kept[-1][0], max(end, kept[-1][1]))
        else:
            kept.append((start, end))
    return [text[start:end] for start, end in kept]


def is_initials(name: str) -> bool:
    """
    Checks whether a name the scrubber found is only initials, such as a staff member's in a records column.

    Initials are kept: they don't identify anyone outside the clinic. A first or last name is scrubbed, even a short
    one, unless it's two letters written in capitals ("AL"), which can't be told apart from initials.

    Parameters
    ----------
    name : str
        The name's text.

    Returns
    -------
    bool
        Whether it's only initials.

    Examples
    --------
    >>> [is_initials(name) for name in ['JO', 'J.O.', 'J. M. O.', ' KB\\n']]
    [True, True, True, True]
    >>> [is_initials(name) for name in ['Jo', 'Ortiz', 'TOM', 'Dr. AW', 'MRS. COX', 'J Ortiz', 'Al']]
    [False, False, False, False, False, False, False]
    """
    return bool(_INITIALS_RE.fullmatch(name.strip()))


# Month names that are also first names. A date the model found that is only one of these ("Spoke with April") may
# be a name, and a date that's only a month name isn't one the visit date miner can use, so it's scrubbed.
_NAME_LIKE_MONTH_RE = re.compile(r'(?:jan|april|may|june|august)\.?', re.IGNORECASE)


def is_name_like_month(text: str) -> bool:
    """
    Checks whether a date the scrubber found is only a month name that is also a first name, so it's scrubbed
    rather than kept.

    Parameters
    ----------
    text : str
        The date's text.

    Returns
    -------
    bool
        Whether it's only a month name that is also a first name.

    Examples
    --------
    >>> [is_name_like_month(text) for text in ['April', ' JUNE ', 'May', 'Jan.']]
    [True, True, True, True]
    >>> [is_name_like_month(text) for text in ['April 3', 'March', 'Monday', 'June 2023', 'yesterday']]
    [False, False, False, False, False]
    """
    return bool(_NAME_LIKE_MONTH_RE.fullmatch(text.strip()))


def is_date_part(text: str) -> bool:
    """
    Checks whether text the scrubber found inside a date is only part of the date, such as "June" read as a name in
    "June 3, 2023", so it can be kept with the date.

    Parameters
    ----------
    text : str
        The detection's text.

    Returns
    -------
    bool
        Whether it's made only of month and weekday names and a date's numbers.

    Examples
    --------
    >>> [is_date_part(text) for text in ['June', 'June 3', 'Tues.', '3rd', '2023', '3/14/2023', '03-14-23', '031423']]
    [True, True, True, True, True, True, True, True]
    >>> [is_date_part(text) for text in ['Jennifer', '094-012345', '555-0134', '541', 'Bella', '12345', '331423']]
    [False, False, False, False, False, False, False]
    """
    return bool(_DATE_PART_RE.fullmatch(text.strip()))
