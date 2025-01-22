import json
from datetime import datetime
from typing import Any, List, Optional

from dateutil import parser

from vmrt_tesseract_utilities.logging import stdout_logger

"""
Extracts dates from a specified file.
"""


def find_dates(data: Any, min_date: Optional[datetime] = None, max_date: Optional[datetime] = None) -> List[datetime]:
    """
    Recursively searches for and extracts date values from a data structure.

    Parameters
    ----------
    data : str, list, or dict
        The data structure to search.
    min_date : datetime, optional
        The earliest date to include in the results. Defaults to the `min_date`
        provided to the `DateExtractor` instance, or 2010-01-01 if not specified.
    max_date : datetime, optional
        The latest date to include in the results. Defaults to the `max_date`
        provided to the `DateExtractor` instance, or the current date and time if not specified.

    Returns
    -------
    List[datetime]
        A list of datetime objects representing the dates found.
    """

    # Use instance's min_date and max_date if not explicitly provided
    min_date = datetime.strptime('2010', '%Y') if min_date is None else min_date
    max_date = datetime.now() if max_date is None else max_date

    dates_found = []
    if isinstance(data, str):
        try:
            date_object = parser.parse(data, ignoretz=True, fuzzy=True)
            if min_date <= date_object <= max_date:
                dates_found.append(date_object)
        except Exception:
            pass
    elif isinstance(data, list):
        for item in data:
            dates_found.extend(find_dates(item, min_date, max_date))
    elif isinstance(data, dict):
        for value in data.values():
            dates_found.extend(find_dates(value, min_date, max_date))
    return dates_found


class DateExtractor:
    """
    Extracts dates from a specified file.
    """

    def __init__(self, filepath: str, min_date: Optional[datetime] = None, max_date: Optional[datetime] = None):
        """
        Initializes DateExtractor with a file path and optional date range.

        Parameters
        ----------
        filepath : str
            The path to the file.
        min_date : datetime, optional
            The earliest date to include in the results (default is None).
        max_date : datetime, optional
            The latest date to include in the results (default is None).
        """

        self.filepath = filepath
        self.min_date = min_date
        self.max_date = max_date

    def extract_dates_from_file(self) -> List[datetime]:
        """
        Extracts dates from the file.

        Returns
        -------
        List[datetime]
            A list of datetime objects representing the dates found in the file.
        """

        dates_found = []
        if self.filepath.endswith('.json'):
            with open(self.filepath, 'r') as f:
                try:
                    data = json.load(f)
                    dates_found = find_dates(data, self.min_date, self.max_date)
                except json.JSONDecodeError as e:
                    stdout_logger.error(f'Error decoding JSON in file {self.filepath}: {e}')
        elif self.filepath.endswith('.txt'):
            with open(self.filepath, 'r') as f:
                for line in f:  # Process line by line
                    dates_found.extend(find_dates(line, self.min_date, self.max_date))

        return list(set(dates_found))
