"""
The shape of a GRLS subject ID, in one place.

A subject ID is a 3-digit site prefix, a dash and a 6-digit suffix, e.g. "094-000520". Every pattern that finds,
checks or scrubs subject IDs is built from the constants here, so a change to the ID scheme only needs making once.
"""

import re
from typing import List, Optional

# The study's site prefix. Only IDs with this prefix belong to GRLS dogs.
SITE_PREFIX = '094'

# Digit counts of the two parts of an ID.
PREFIX_LENGTH = len(SITE_PREFIX)
SUFFIX_LENGTH = 6

# OCR'd text sometimes shows a suffix with an extra leading zero ("094-0000520" for "094-000520").
SUFFIX_PADDING = '0'

# [0-9] rather than \d, which also matches non-ASCII digits (e.g. fullwidth "０"), which no real ID contains.
_DIGIT = '[0-9]'

# Checks that a whole string is a subject ID, with any site prefix. Use parse_id() rather than calling .match() or
# .search() on this, which would also accept an ID with extra characters after it.
_ID_RE = re.compile(rf'({_DIGIT}{{{PREFIX_LENGTH}}})-({_DIGIT}{{{SUFFIX_LENGTH}}})')

# Finds a GRLS subject ID inside a longer string, such as a file path. The digit checks on either side stop it
# matching part of a longer number: "094-0001312" is not dog "094-000131".
_SITE_ID_SEARCH_RE = re.compile(rf'(?<!{_DIGIT}){SITE_PREFIX}-{_DIGIT}{{{SUFFIX_LENGTH}}}(?!{_DIGIT})')

# The same, without the digit checks, to spot names that look like a subject ID but aren't accepted as one.
_SITE_ID_LOOSE_RE = re.compile(rf'{SITE_PREFIX}-{_DIGIT}{{{SUFFIX_LENGTH}}}')


def parse_id(text: str) -> Optional[tuple]:
    """
    Splits a string that is exactly one subject ID into its prefix and suffix.

    Parameters
    ----------
    text : str
        The candidate ID.

    Returns
    -------
    tuple of (str, str) or None
        The prefix and suffix, or None if the string is not a subject ID.

    Examples
    --------
    >>> parse_id('094-000520')
    ('094', '000520')
    >>> parse_id('094-0005201x') is None
    True
    """
    match = _ID_RE.fullmatch(text)
    return match.groups() if match else None


def find_subject_id(text: str) -> Optional[str]:
    """
    Finds the first GRLS subject ID (one with the study's site prefix) in a string such as a file path.

    Parameters
    ----------
    text : str
        The string to search.

    Returns
    -------
    str or None
        The subject ID, or None if there isn't one.

    Examples
    --------
    >>> find_subject_id('/data/123-456789/094-000131/record.pdf')
    '094-000131'
    >>> find_subject_id('/data/094-0001312') is None
    True
    """
    match = _SITE_ID_SEARCH_RE.search(text)
    return match.group(0) if match else None


def looks_like_subject_id(text: str) -> bool:
    """
    Checks whether a string contains something shaped like a GRLS subject ID, even one find_subject_id() rejects
    because it touches other digits.

    Parameters
    ----------
    text : str
        The string to check, such as a folder name.

    Returns
    -------
    bool
        Whether the string contains "094-" followed by six digits.

    Examples
    --------
    >>> looks_like_subject_id('094-0001312')
    True
    >>> looks_like_subject_id('2019 visits')
    False
    """
    return bool(_SITE_ID_LOOSE_RE.search(text))


def id_variants(text: str) -> List[str]:
    """
    Lists the forms a subject ID commonly takes in OCR'd text: as written, without the dash ("094000520"), and with
    an extra zero-padded digit in the suffix ("094-0000520", "0940000520").

    Parameters
    ----------
    text : str
        A string that is exactly one subject ID (see parse_id()).

    Returns
    -------
    list of str
        The four variants, which are always distinct, or an empty list if the string is not a subject ID.

    Examples
    --------
    >>> id_variants('094-000520')
    ['094-000520', '094000520', '094-0000520', '0940000520']
    """
    parts = parse_id(text)
    if not parts:
        return []
    prefix, suffix = parts
    padded_suffix = f'{SUFFIX_PADDING}{suffix}'
    return [f'{prefix}-{suffix}', f'{prefix}{suffix}', f'{prefix}-{padded_suffix}', f'{prefix}{padded_suffix}']


def ocr_subject_id_regex() -> str:
    """
    Builds a regex that finds GRLS subject IDs in OCR'd text, for the PII scrubber.

    As well as the variants id_variants() lists, it allows a space or en dash in place of the dash and a letter O
    read for the leading zero of the site prefix. \\d is used deliberately here: when scrubbing, matching too much is
    safer than missing an ID.

    Returns
    -------
    str
        The regex.

    Examples
    --------
    >>> ocr_subject_id_regex()
    '(?<!\\\\d)[0O]94[-– ]?0?\\\\d{6}(?!\\\\d)'
    """
    first, rest = SITE_PREFIX[0], SITE_PREFIX[1:]
    leading = f'[{first}O]' if first == '0' else first
    return rf'(?<!\d){leading}{rest}[-– ]?{SUFFIX_PADDING}?\d{{{SUFFIX_LENGTH}}}(?!\d)'
