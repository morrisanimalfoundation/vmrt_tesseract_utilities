"""
Generates PDFs of made-up text for the OCR tests, the Docker smoke test and the end-to-end pipeline test.

A `SyntheticPage` is a scanned page: drawn with Pillow in the DejaVu Sans font, which the workspace image includes,
at US Letter size and 200 dpi, it's only an image. It can be rotated, as if scanned sideways or upside down.

A `DigitalPage` is a born-digital page, as printed to PDF by clinic software: its text is real text, in Helvetica,
that `pdftotext` copies out exactly. It can also have a scanned image on it, have its text drawn invisibly (as a
scanner's own OCR layer over a scan), or have a broken font encoding, so it looks right but copies out as symbols.

In both, a line is either a string or a tuple of table cells, drawn in fixed columns. Never put real record text in a
page: all test data must be made up.

    from tests.synthetic_pdfs import DigitalPage, SyntheticPage, write_pdf

    write_pdf('example.pdf', [SyntheticPage(['Vaccines current', 'Weight 32 kg']), DigitalPage(['Rabies given'])])
"""

import io
import string
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
# US Letter at the resolution below.
PAGE_SIZE = (1700, 2200)
RESOLUTION = 200
LEFT_MARGIN = 150
TOP_MARGIN = 200
# Where each table cell starts, left to right.
COLUMN_POSITIONS = (150, 650, 1150)

Line = Union[str, Tuple[str, ...]]


@dataclass
class SyntheticPage:
    """
    One page of a synthetic PDF.

    Attributes
    ----------
    lines : Sequence[Line]
        The lines of text, top to bottom. A tuple is one table row, with a cell in each column.
    font_size : int
        The font size in pixels.
    rotation : int
        Degrees to turn the finished page counter-clockwise: 0, 90, 180 or 270.
    """
    lines: Sequence[Line]
    font_size: int = 48
    rotation: int = 0

    @property
    def text(self) -> str:
        """
        The page's text as one string, with table cells separated by spaces and lines by newlines.

        >>> SyntheticPage(['Exam', ('Rabies', '03/14/2023')]).text
        'Exam\\nRabies 03/14/2023'

        The checks in `render` stop a test from expecting text that isn't on the page:

        >>> SyntheticPage([('a', 'b', 'c', 'd')]).render()
        Traceback (most recent call last):
            ...
        ValueError: Table row ('a', 'b', 'c', 'd') has more cells than the 3 columns.
        """
        return '\n'.join(' '.join(line) if isinstance(line, tuple) else line for line in self.lines)

    def render(self) -> Image.Image:
        """
        Draws the page.

        Returns
        -------
        Image.Image
            The page image, landscape if rotated by 90 or 270 degrees.

        Raises
        ------
        ValueError
            If some of the text wouldn't be drawn (a table row with more cells than columns, or text running off the
            page), since `text` would then include words the page doesn't have.
        """
        image = Image.new('RGB', PAGE_SIZE, 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, self.font_size)
        line_height = int(self.font_size * 1.6)
        for i, line in enumerate(self.lines):
            y = TOP_MARGIN + i * line_height
            if y + line_height > PAGE_SIZE[1]:
                raise ValueError(f'Line {i + 1} ({line!r}) runs off the bottom of the page.')
            if isinstance(line, tuple):
                if len(line) > len(COLUMN_POSITIONS):
                    raise ValueError(f'Table row {line!r} has more cells than the {len(COLUMN_POSITIONS)} columns.')
                cells = list(zip(COLUMN_POSITIONS, line))
            else:
                cells = [(LEFT_MARGIN, line)]
            for column, (x, cell) in enumerate(cells):
                # A cell ends where the next column starts, and the last one at the page edge.
                right = COLUMN_POSITIONS[column + 1] if isinstance(line, tuple) and column + 1 < len(cells) \
                    else PAGE_SIZE[0]
                if x + font.getlength(cell) > right:
                    raise ValueError(f'{cell!r} is too wide for its space on the page.')
                draw.text((x, y), cell, fill='black', font=font)
        if self.rotation:
            image = image.rotate(self.rotation, expand=True, fillcolor='white')
        return image


# A born-digital page: US Letter in points, with text from the top-left margin.
POINTS_PER_INCH = 72
DIGITAL_PAGE_SIZE = (612, 792)
DIGITAL_LEFT_MARGIN = 72
DIGITAL_TOP = 720
# Where each table cell starts, left to right, in points.
DIGITAL_COLUMN_POSITIONS = (72, 252, 432)
# A broken encoding's stand-ins for letters: what a real broken page copied out as was mostly ASCII punctuation.
GARBLED_STAND_INS = '!#$%&*+<=>?@^~|;'
# Helvetica's character widths, in thousandths of the font size, for ASCII from the space to "~".
HELVETICA_WIDTHS = dict(zip(
    [chr(c) for c in range(32, 127)],
    [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556,
     556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667,
     556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556,
     556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722,
     500, 500, 500, 334, 260, 334, 584]))


@dataclass
class DigitalPage:
    """
    One born-digital page of a synthetic PDF.

    Attributes
    ----------
    lines : Sequence[Line]
        The lines of text, top to bottom, in plain ASCII. A tuple is one table row, with a cell in each column.
    font_size : int
        The font size in points.
    image : SyntheticPage, optional
        A scanned page to draw on it, behind the text.
    image_box : tuple[float, float, float, float], optional
        Where to draw the image, as (x, y, width, height) in points from the bottom-left corner. The whole page if
        not given.
    image_strips : int
        Store the image as this many horizontal strips, as some scanners and fax software do.
    inline_image : bool
        Store the image inside the page's drawing instructions (an inline image) rather than as its own object.
    invisible : bool
        Draw the text invisibly, as a scanner's or fax's OCR layer over a scan: pdftotext still copies it out.
    garbled : bool
        Give the font a broken encoding: the page looks right, but its letters copy out as punctuation.
    """
    lines: Sequence[Line]
    font_size: int = 12
    image: Optional[SyntheticPage] = None
    image_box: Optional[Tuple[float, float, float, float]] = None
    image_strips: int = 1
    inline_image: bool = False
    invisible: bool = False
    garbled: bool = False

    @property
    def text(self) -> str:
        """
        The page's text as one string, with table cells separated by spaces and lines by newlines.

        >>> DigitalPage(['Exam', ('Rabies', '03/14/2023')]).text
        'Exam\\nRabies 03/14/2023'
        """
        return '\n'.join(' '.join(line) if isinstance(line, tuple) else line for line in self.lines)

    def text_content(self) -> bytes:
        """
        The drawing instructions for the page's text, in font /F1.

        Raises
        ------
        ValueError
            If some of the text wouldn't be on the page, since `text` would then include words the page doesn't have:
            a line that isn't plain ASCII, a table row with more cells than columns, or text running off the page or
            into the next column.

        >>> DigitalPage([('a', 'b', 'c', 'd')]).text_content()
        Traceback (most recent call last):
            ...
        ValueError: Table row ('a', 'b', 'c', 'd') has more cells than the 3 columns.
        >>> DigitalPage(['Vaccines current. ' * 6]).text_content()
        Traceback (most recent call last):
            ...
        ValueError: 'Vaccines current. Vaccines current. ...' is too wide for its space on the page.
        """
        line_height = self.font_size * 1.5
        ops = [f'BT /F1 {self.font_size} Tf {3 if self.invisible else 0} Tr']
        for i, line in enumerate(self.lines):
            y = DIGITAL_TOP - i * line_height
            if y < POINTS_PER_INCH / 2:
                raise ValueError(f'Line {i + 1} ({line!r}) runs off the bottom of the page.')
            if isinstance(line, tuple):
                if len(line) > len(DIGITAL_COLUMN_POSITIONS):
                    raise ValueError(f'Table row {line!r} has more cells than the {len(DIGITAL_COLUMN_POSITIONS)} '
                                     'columns.')
                cells = list(zip(DIGITAL_COLUMN_POSITIONS, line))
            else:
                cells = [(DIGITAL_LEFT_MARGIN, line)]
            for column, (x, cell) in enumerate(cells):
                if not cell.isascii() or not cell.isprintable():
                    raise ValueError(f'{cell!r} is not plain ASCII.')
                # A cell ends where the next column starts, and the last one at the page edge.
                right = DIGITAL_COLUMN_POSITIONS[column + 1] if column + 1 < len(cells) else DIGITAL_PAGE_SIZE[0]
                if x + sum(HELVETICA_WIDTHS[c] for c in cell) * self.font_size / 1000 > right:
                    shown = cell if len(cell) <= 40 else cell[:36] + '...'
                    raise ValueError(f'{shown!r} is too wide for its space on the page.')
                escaped = cell.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
                ops.append(f'1 0 0 1 {x} {y:g} Tm ({escaped}) Tj')
        ops.append('ET')
        return '\n'.join(ops).encode('ascii')


def garbled_to_unicode() -> bytes:
    """
    A ToUnicode map that copies each letter out as a punctuation mark, as a broken font encoding does.
    """
    letters = string.ascii_letters
    pairs = ''.join(f'<{ord(c):02X}> <{ord(GARBLED_STAND_INS[i % len(GARBLED_STAND_INS)]):04X}>\n'
                    for i, c in enumerate(letters))
    return (f'/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n'
            f'/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n'
            f'/CMapName /Garbled def /CMapType 2 def\n'
            f'1 begincodespacerange <00> <FF> endcodespacerange\n'
            f'{len(letters)} beginbfchar\n{pairs}endbfchar\n'
            f'endcmap CMapName currentdict /CMap defineresource pop end end\n').encode('ascii')


def write_pdf(path: str, pages: Sequence[Union[SyntheticPage, DigitalPage]]) -> str:
    """
    Writes the pages to a PDF.

    A PDF of only scanned pages is saved by Pillow, as it always has been, so OCR tests read the same images. One
    with a born-digital page is written directly, with each scanned page as a full-page JPEG.

    Parameters
    ----------
    path : str
        Where to write the PDF.
    pages : Sequence[SyntheticPage | DigitalPage]
        The pages, in order.

    Returns
    -------
    str
        The path, for convenience.
    """
    if all(isinstance(page, SyntheticPage) for page in pages):
        images = [page.render() for page in pages]
        images[0].save(path, 'PDF', resolution=RESOLUTION, save_all=True, append_images=images[1:])
        return path
    with open(path, 'wb') as f:
        f.write(_digital_pdf(pages))
    return path


def _jpeg(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=95)
    return buffer.getvalue()


def _digital_pdf(pages: Sequence[Union[SyntheticPage, DigitalPage]]) -> bytes:
    """
    Builds a PDF with born-digital pages (and maybe scanned ones), object by object.
    """
    objects = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    def stream(dictionary: str, data: bytes) -> bytes:
        return f'<< {dictionary} /Length {len(data)} >>\nstream\n'.encode('ascii') + data + b'\nendstream'

    catalog = add(b'')
    page_tree = add(b'')
    font = add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>')
    to_unicode = add(stream('', garbled_to_unicode()))
    garbled_font = add(f'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding '
                       f'/ToUnicode {to_unicode} 0 R >>'.encode('ascii'))
    page_ids = []
    for page in pages:
        size = DIGITAL_PAGE_SIZE
        if isinstance(page, SyntheticPage):
            # A scanned page: the image fills a page of its own size, so a turned page is landscape.
            image = page.render()
            size = (image.width * POINTS_PER_INCH / RESOLUTION, image.height * POINTS_PER_INCH / RESOLUTION)
            image.close()
            page = DigitalPage([], image=page, image_box=(0, 0) + size)
        image_ops, xobjects = [], []
        if page.image is not None:
            image = page.image.render()
            try:
                x, y, width, height = page.image_box or (0, 0) + DIGITAL_PAGE_SIZE
                for strip in range(page.image_strips):
                    # Strips go top to bottom; PDF's y goes up from the bottom of the page.
                    top, bottom = (image.height * k // page.image_strips for k in (strip, strip + 1))
                    piece = image.crop((0, top, image.width, bottom))
                    data = _jpeg(piece)
                    strip_height = height * (bottom - top) / image.height
                    strip_y = y + height * (image.height - bottom) / image.height
                    place = f'q {width:g} 0 0 {strip_height:g} {x:g} {strip_y:g} cm'
                    if page.inline_image:
                        image_ops.append(f'{place} BI /W {piece.width} /H {piece.height} /CS /RGB /BPC 8 '
                                         f'/F [/AHx /DCT] ID\n{data.hex()}>\nEI Q'.encode('ascii'))
                    else:
                        name = f'Im{len(xobjects) + 1}'
                        xobjects.append(f'/{name} ' + str(add(stream(
                            f'/Type /XObject /Subtype /Image /Width {piece.width} /Height {piece.height} '
                            f'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode', data))) + ' 0 R')
                        image_ops.append(f'{place} /{name} Do Q'.encode('ascii'))
                    piece.close()
            finally:
                image.close()
        resources = f'/Font << /F1 {garbled_font if page.garbled else font} 0 R >>'
        if xobjects:
            resources += f' /XObject << {" ".join(xobjects)} >>'
        content = add(stream('', b'\n'.join(image_ops + [page.text_content()])))
        page_ids.append(add(f'<< /Type /Page /Parent {page_tree} 0 R /MediaBox [0 0 {size[0]:g} {size[1]:g}] '
                            f'/Resources << {resources} >> /Contents {content} 0 R >>'.encode('ascii')))
    objects[catalog - 1] = f'<< /Type /Catalog /Pages {page_tree} 0 R >>'.encode('ascii')
    kids = ' '.join(f'{i} 0 R' for i in page_ids)
    objects[page_tree - 1] = f'<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>'.encode('ascii')

    out = bytearray(b'%PDF-1.4\n')
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f'{number} 0 obj\n'.encode('ascii') + body + b'\nendobj\n'
    xref = len(out)
    out += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode('ascii')
    out += ''.join(f'{offset:010d} 00000 n \n' for offset in offsets).encode('ascii')
    out += f'trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode('ascii')
    return bytes(out)
