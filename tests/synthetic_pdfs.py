"""
Generates PDFs of made-up text for the OCR tests, the Docker smoke test and the end-to-end pipeline test.

Pages are drawn with Pillow in the DejaVu Sans font, which the workspace image includes, at US Letter size and 200
dpi, like a scanned record. A line is either a string or a tuple of table cells, drawn in fixed columns. Pages can
be rotated, as if scanned sideways or upside down. Never put real record text in a page: all test data must be made
up.

    from tests.synthetic_pdfs import SyntheticPage, write_pdf

    write_pdf('example.pdf', [SyntheticPage(['Vaccines current', 'Weight 32 kg'])])
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

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
        """
        return '\n'.join(' '.join(line) if isinstance(line, tuple) else line for line in self.lines)

    def render(self) -> Image.Image:
        """
        Draws the page.

        Returns
        -------
        Image.Image
            The page image, landscape if rotated by 90 or 270 degrees.
        """
        image = Image.new('RGB', PAGE_SIZE, 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, self.font_size)
        line_height = int(self.font_size * 1.6)
        for i, line in enumerate(self.lines):
            y = TOP_MARGIN + i * line_height
            cells = line if isinstance(line, tuple) else (line,)
            for x, cell in zip(COLUMN_POSITIONS if isinstance(line, tuple) else (LEFT_MARGIN,), cells):
                draw.text((x, y), cell, fill='black', font=font)
        if self.rotation:
            image = image.rotate(self.rotation, expand=True, fillcolor='white')
        return image


def write_pdf(path: str, pages: Sequence[SyntheticPage]) -> str:
    """
    Writes the pages to a PDF.

    Parameters
    ----------
    path : str
        Where to write the PDF.
    pages : Sequence[SyntheticPage]
        The pages, in order.

    Returns
    -------
    str
        The path, for convenience.
    """
    images = [page.render() for page in pages]
    images[0].save(path, 'PDF', resolution=RESOLUTION, save_all=True, append_images=images[1:])
    return path
