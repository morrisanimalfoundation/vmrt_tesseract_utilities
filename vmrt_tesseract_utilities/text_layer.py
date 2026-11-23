"""
Reads the text a PDF already holds (its text layer) and decides, page by page, whether that text can be used instead
of OCR.

Most EMR pages were made on a computer, so the PDF holds their exact text: `pdftotext` copies it out with no reading
mistakes, in a fraction of the time OCR takes. A page still needs OCR when:

- it has no text layer, or too little to be the page's text (a fax header or page number added to a scan);
- it's a scan: an image covers most of it. A scanner's or fax's own OCR may have added a text layer, but its quality
  is unknown, so it's read again;
- an image covers a sizeable part of it, such as a scanned lab report or receipt pasted into a digital page: the text
  layer has none of the image's text;
- its text layer is garbled: a broken font encoding can make text that looks right on the page copy out as symbols.

The cut-offs below were measured on the real records (230,709 pages in 20,762 PDFs, November 2026); see issue #82.
"""

import re
import subprocess
import unicodedata
from dataclasses import dataclass

# A page's text layer is its text when it has at least this many characters that aren't spaces. A fax header or a
# page number added to a scan is usually under it.
MIN_TEXT_CHARS = 100
# Images covering at least this share of the page, added up, make it a scan: a scan can be stored as strips or tiles.
SCAN_COVERAGE = 0.5
# An image covering at least this share of the page may hold text the text layer doesn't have, so the page is read
# with OCR. On real born-digital pages, a quarter to a third of those with an image of 10-50% of the page had 20 or
# more words in it (up to about 500); with smaller images (logos, signatures) there were as few as with none. It's
# the largest image that counts here, not all of them added up: on 60 real pages whose images only reached 10%
# together, none had more than 12 words in them.
IMAGE_COVERAGE = 0.1
# A text layer is garbled when, with at least MIN_TOKENS_FOR_GARBLED words that have letters in them, fewer than this
# share look like words... This is a loose check: an encoding that swaps letters for other letters can still pass it.
# On 1,000 random pages that pass every check here, none shared under half of its words with OCR (a garbled page
# shares almost none), so that's rare enough here to leave for now...
MIN_WORDLIKE_SHARE = 0.5
MIN_TOKENS_FOR_GARBLED = 20
# ...or when more than this share of its characters are unusual (symbols from a broken font encoding, the
# replacement character)...
MAX_ODD_CHAR_SHARE = 0.2
# ...or when more than this share of its characters are control characters. On a few clean pages pdftotext gives
# about 1-1.7%; checked against OCR, pages over 3% were garbled...
MAX_CONTROL_SHARE = 0.02
# ...or when, leaving out numbers, underscores (form fill-in lines) and repeated runs like "-----" or ". . . .", it has
# more than this many punctuation and symbol characters per letter. That catches a broken encoding that turns letters into
# ASCII punctuation, which the checks above miss. Checked against OCR: clean pages had up to 0.21, garbled ones 0.6
# and up.
MAX_SYMBOLS_PER_LETTER = 0.4
# Seconds one poppler command may take on one file.
TOOL_TIMEOUT = 300
# More pages than any PDF will have.
MAX_PAGES = 1_000_000

# Characters that count as usual: ASCII, plus Latin-1 letters and symbols such as °, µ and ±, and typographic
# dashes, quotes and bullets.
USUAL_CHAR_RE = re.compile(r'[\x20-\x7e\xa0-\xff‐-‧]')
NUMBER_RE = re.compile(r'\d+(?:[.,/:-]\d+)*')
WORD_RE = re.compile(r'[A-Za-z]+(?:[\'-][A-Za-z]+)*')
VOWEL_RE = re.compile(r'[aeiouyAEIOUY]')
# Punctuation that can surround a word, including typographic quotes and the degree sign ("°F").
EDGE_PUNCTUATION = '.,:;!?()[]{}"\'*/#\u2018\u2019\u201c\u201d\u00b0'

# Why a page needs OCR (PageText.ocr_reason).
NO_TEXT = 'no text layer'
UNKNOWN_SIZE = 'unknown page size'
SCAN = 'scan'
IMAGE = 'large image'
GARBLED = 'garbled text layer'


class TextLayerError(Exception):
    """
    Raised when poppler can't read a PDF (it's damaged, needs a password, or takes too long).
    """
    pass


@dataclass
class PageText:
    """
    One page's text layer, and whether it can be used instead of OCR.

    Attributes
    ----------
    page : int
        The page number, from 1.
    text : str
        The page's text from `pdftotext -layout`, which keeps each table row on one line. Empty if it has none.
    ocr_reason : str
        Why the page needs OCR (NO_TEXT, UNKNOWN_SIZE, SCAN, IMAGE, or GARBLED with the reason after a colon), or ''
        if its text layer can be used.
    """
    page: int
    text: str
    ocr_reason: str

    @property
    def use_text_layer(self) -> bool:
        return not self.ocr_reason


def read_pages(input_path: str) -> list[PageText]:
    """
    Reads every page's text layer and decides whether it can be used instead of OCR.

    Parameters
    ----------
    input_path : str
        The path to the PDF.

    Returns
    -------
    list[PageText]
        One per page, in order.

    Raises
    ------
    TextLayerError
        If pdfinfo, pdfimages or pdftotext fails on the file.
    """
    # pdfinfo stops at the last page, so a last page past it lists every page's size in one call.
    areas = page_areas(_run(['pdfinfo', '-f', '1', '-l', str(MAX_PAGES), input_path], input_path))
    images = image_areas(_run(['pdfimages', '-list', input_path], input_path), areas)
    texts = split_pages(_run(['pdftotext', '-layout', '-enc', 'UTF-8', input_path, '-'], input_path),
                        len(areas))
    return [PageText(page, text, ocr_reason(text, area, largest, total))
            for page, (text, area, (largest, total)) in enumerate(zip(texts, areas, images), 1)]


def ocr_reason(text: str, page_area: float, largest_image: float, all_images: float) -> str:
    """
    Says why a page needs OCR, or '' if its text layer can be used.

    Parameters
    ----------
    text : str
        The page's text layer.
    page_area : float
        The page's area, in square inches.
    largest_image : float
        The area of the largest image on it, in square inches.
    all_images : float
        The area all its images cover, in square inches (see `image_areas`).

    >>> page = 'Annual exam. Weight 32 kg. Vaccines current. ' * 3
    >>> ocr_reason(page, 93.5, 0, 0)
    ''
    >>> ocr_reason('Page 1 of 3', 93.5, 0, 0)
    'no text layer'
    >>> ocr_reason(page, 0, 0, 0)
    'unknown page size'
    >>> ocr_reason(page, 93.5, 90, 90), ocr_reason(page, 93.5, 20, 20), ocr_reason(page, 93.5, 3, 3)
    ('scan', 'large image', '')

    A scan stored as strips is still a scan, but several small images (a logo, a signature) aren't a large one:

    >>> ocr_reason(page, 93.5, 6, 93.5), ocr_reason(page, 93.5, 6, 18)
    ('scan', '')
    >>> ocr_reason('%$#& ' * 30, 93.5, 0, 0)
    'garbled text layer: symbols'
    """
    chars = re.sub(r'\s', '', text)
    if len(chars) < MIN_TEXT_CHARS:
        return NO_TEXT
    # Without the page's size there's no telling how much of it images cover.
    if page_area <= 0:
        return UNKNOWN_SIZE
    if all_images / page_area >= SCAN_COVERAGE:
        return SCAN
    if largest_image / page_area >= IMAGE_COVERAGE:
        return IMAGE
    reason = garbled_reason(text)
    return f'{GARBLED}: {reason}' if reason else ''


def garbled_reason(text: str) -> str:
    """
    Says why a page's text layer looks like something other than words, or '' when it looks fine.

    >>> garbled_reason('Weight 32 kg. Vaccines current. ' * 10)
    ''
    >>> garbled_reason('White blood cells 6.5 x10^3/uL (5.5-16.9) ---------- ' * 10)
    ''
    >>> garbled_reason('Owner: ____ ____ Date: __/__/__ Signed: _____ ' * 10)
    ''
    >>> garbled_reason('Exam fee . . . . . . . . . . . . . . . . . . $45.00\\n' * 10)
    ''
    >>> garbled_reason('\\ufffd\\ufffd ' * 50)
    'odd characters'
    >>> garbled_reason('a80 %$ \\x07(!# Clinic ' * 20)
    'control characters'
    >>> garbled_reason('a80 %$ (!#& ;<>% a113 Clinic ' * 20)
    'symbols'
    >>> garbled_reason('xkcdzt prwqlm ' * 20)
    'not words'
    """
    chars = re.sub(r'\s', '', text)
    if not chars:
        return ''
    if sum(1 for c in chars if not USUAL_CHAR_RE.match(c)) / len(chars) > MAX_ODD_CHAR_SHARE:
        return 'odd characters'
    if sum(1 for c in chars if unicodedata.category(c) == 'Cc') / len(chars) > MAX_CONTROL_SHARE:
        return 'control characters'
    rest = re.sub(r'([^\w\s])(?: ?\1)+', r'\1', NUMBER_RE.sub(' ', text).replace('_', ' '))
    letters = sum(c.isalpha() for c in rest)
    symbols = sum(1 for c in rest if not c.isspace() and not c.isalnum())
    if symbols > MAX_SYMBOLS_PER_LETTER * letters:
        return 'symbols'
    tokens = [t for t in text.split() if re.search(r'[A-Za-z]', t)]
    if len(tokens) >= MIN_TOKENS_FOR_GARBLED and sum(map(is_wordlike, tokens)) / len(tokens) < MIN_WORDLIKE_SHARE:
        return 'not words'
    return ''


def is_wordlike(token: str) -> bool:
    """
    Tells whether a token with letters in it looks like a word or an abbreviation.

    >>> [is_wordlike(t) for t in ('Rabies', 'mg', 'CBC', "dog's", '\u201cnormal\u201d', '\u00b0F', 'q.i.d.', 'xkcdzt',
    ...                           'a7#f', '\u25a1\u25a1')]
    [True, True, True, True, True, True, False, False, False, False]
    """
    word = token.strip(EDGE_PUNCTUATION)
    if not WORD_RE.fullmatch(word):
        return False
    return len(word) <= 3 or word.isupper() or bool(VOWEL_RE.search(word))


def page_areas(pdfinfo_output: str) -> list[float]:
    """
    Reads each page's area, in square inches, from the output of `pdfinfo -f 1 -l N`.

    The size pdfinfo gives is the page's CropBox, the part that's shown.

    >>> page_areas('Pages:          2\\nPage    1 size: 612 x 792 pts (letter)\\nPage    2 size: 306 x 396 pts\\n')
    [93.5, 23.375]
    """
    match = re.search(r'^Pages:\s+(\d+)', pdfinfo_output, re.MULTILINE)
    if not match:
        raise TextLayerError('pdfinfo gave no page count.')
    areas = [0.0] * int(match.group(1))
    for page, width, height in re.findall(r'^Page\s+(\d+) size:\s+([\d.]+) x ([\d.]+) pts', pdfinfo_output,
                                          re.MULTILINE):
        if 1 <= int(page) <= len(areas):
            areas[int(page) - 1] = float(width) / 72 * float(height) / 72
    return areas


def image_areas(pdfimages_output: str, areas: list[float]) -> list[tuple[float, float]]:
    """
    Reads, from the output of `pdfimages -list`, the area of the largest image on each page and of all its images
    added up (up to the page's own area), in square inches.

    An image's size on the page is its size in pixels divided by its resolution there. An image with under 1 pixel
    per inch along a side (pdfimages shows 0) is left out: it can't hold readable text or be a scan, and PDF makers
    use stretched images of a pixel or two for rules, borders and background fills.

    Parameters
    ----------
    pdfimages_output : str
        What `pdfimages -list` printed.
    areas : list[float]
        Each page's area, in square inches (see `page_areas`).

    >>> header = ('page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi size ratio\\n'
    ...           '--------------------------------------------------------------------------------------------\\n')
    >>> image_areas(header +
    ...             '   1     0 image    1700  2200  rgb     3   8  jpeg   no         7  0   200   200  263K 2.4%\\n'
    ...             '   1     1 smask    1700  2200  gray    1   8  image  no         7  0   200   200  1K 0.1%\\n'
    ...             '   2     2 image     300   300  rgb     3   8  jpeg   no         9  0   300   300  10K 3.0%\\n'
    ...             '   2     3 image     300   300  rgb     3   8  jpeg   no         9  0   300   300  10K 3.0%\\n',
    ...             [93.5, 93.5, 93.5])
    [(93.5, 93.5), (1.0, 2.0), (0.0, 0.0)]

    An inline image has one column for its object ID, "[inline]", instead of two:

    >>> image_areas(header + '   1     0 image     850  1100  gray    1   8  image  no   [inline]   100   100  0B 0.0%\\n',
    ...             [93.5])
    [(93.5, 93.5)]
    >>> image_areas(header + '   1     0 image       1     1  gray    1   8  image  no         7  0     0    72  1B 0.0%\\n',
    ...             [93.5])
    [(0.0, 0.0)]
    """
    largest, covered = [0.0] * len(areas), [0.0] * len(areas)
    for line in pdfimages_output.splitlines()[2:]:
        cols = line.split()
        # page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio. "object ID" is two
        # columns ("7  0"), or one for an inline image ("[inline]"), so x-ppi and y-ppi are counted from the end.
        if len(cols) < 15 or not cols[0].isdigit() or cols[2] not in ('image', 'stencil'):
            continue
        try:
            page, width, height = int(cols[0]), int(cols[3]), int(cols[4])
            x_ppi, y_ppi = float(cols[-4]), float(cols[-3])
        except ValueError:
            continue
        if not 1 <= page <= len(areas):
            continue
        if x_ppi <= 0 or y_ppi <= 0:
            continue
        area = width / x_ppi * height / y_ppi
        largest[page - 1] = max(largest[page - 1], area)
        covered[page - 1] = min(covered[page - 1] + area, areas[page - 1])
    return list(zip(largest, covered))


def split_pages(pdftotext_output: str, count: int) -> list[str]:
    """
    Splits pdftotext's output into pages: it ends each page with a form feed.

    >>> split_pages('first\\n\\fsecond\\n\\f', 3)
    ['first\\n', 'second\\n', '']
    """
    pages = pdftotext_output.split('\f')[:count]
    return pages + [''] * (count - len(pages))


def _run(args: list[str], input_path: str) -> str:
    """
    Runs a poppler command on a PDF and returns what it printed.

    Raises
    ------
    TextLayerError
        If it can't be run, fails, or takes longer than TOOL_TIMEOUT.
    """
    try:
        done = subprocess.run(args, capture_output=True, timeout=TOOL_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise TextLayerError(f'{args[0]} took longer than {TOOL_TIMEOUT} seconds on {input_path}.')
    except OSError as e:
        # Poppler isn't installed (outside the Docker image) or can't be run.
        raise TextLayerError(f'{args[0]} could not be run on {input_path}: {e}') from e
    if done.returncode != 0:
        error = done.stderr.decode('utf-8', 'replace').strip()[:500]
        raise TextLayerError(f'{args[0]} exited with code {done.returncode} on {input_path}: {error}')
    # A broken encoding can give bytes that aren't UTF-8; they become U+FFFD, which garbled_reason counts as odd.
    return done.stdout.decode('utf-8', 'replace')
