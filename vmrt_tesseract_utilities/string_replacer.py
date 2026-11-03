import re
from typing import List

from vmrt_tesseract_utilities.grls_id import id_variants, parse_id

"""
Provides string replacement functionality.
"""


class StringReplacer:
    """
    Replaces target strings within a text blob with a specified replacement string.

    A target string that looks like a GRLS subject ID (e.g. "094-000520") is also
    matched in the variant forms it commonly appears as in OCR'd text: without the
    dash ("094000520"), and with an extra zero-padded digit in the suffix
    ("094-0000520", "0940000520").

    Parameters
    ----------
    target_strings : list of str
        The strings to be replaced.
    replacement_string : str
        The string to replace the target strings with.

    Examples
    --------
    >>> replacer = StringReplacer(["old_string1", "old_string2"], "<new_string>")
    >>> text_blob = "This is a text blob with old_string1 and old_string2."
    >>> new_blob = replacer.replace(text_blob)
    >>> print(new_blob)
    This is a text blob with <new_string> and <new_string>.
    """
    def __init__(self, target_strings: List[str], replacement_string: str):
        """
        Initializes the StringReplacer.

        Parameters
        ----------
        target_strings : list of str
            The strings to be replaced.
        replacement_string : str
            The string to replace the target strings with.
        """
        # Blank entries are dropped: an empty pattern matches at every position in
        # the text, so a blank target would insert the replacement between every
        # character instead of doing nothing.
        self.target_strings = [target for target in target_strings if target and target.strip()]
        self.replacement_string = replacement_string
        self._patterns = [self._build_pattern(target) for target in self.target_strings]

    @staticmethod
    def _expand_id_variants(target: str) -> List[str]:
        """
        Expands a GRLS-style subject ID into the variant forms it may appear as in
        OCR'd text. Strings that don't look like a GRLS ID are returned unchanged.

        Parameters
        ----------
        target : str
            The candidate target string.

        Returns
        -------
        list of str
            The target string, plus any additional ID variants.
        """
        return id_variants(target) or [target]

    @classmethod
    def _build_pattern(cls, target: str) -> re.Pattern:
        """
        Builds a single compiled pattern matching a target string and all of its
        ID variants, so the text only needs to be scanned once per target.

        Parameters
        ----------
        target : str
            The target string to build a pattern for.

        Returns
        -------
        re.Pattern
            A compiled, case-insensitive pattern matching any variant of the target.
        """
        is_id = parse_id(target) is not None
        # Longest first, so a variant that's a prefix of another isn't matched short.
        variants = sorted(cls._expand_id_variants(target), key=len, reverse=True)
        alternation = '|'.join(re.escape(variant) for variant in variants)
        if is_id:
            # ID variants start and end with digits; without a boundary check, one
            # could match inside an unrelated, longer digit run (a fax number, a
            # timestamp) rather than only as a standalone ID.
            alternation = rf'(?<!\d)(?:{alternation})(?!\d)'
        return re.compile(alternation, flags=re.IGNORECASE)

    def replace(self, text_blob: str) -> str:
        """
        Replaces the target strings within the text blob.

        Parameters
        ----------
        text_blob : str
            The input text blob.

        Returns
        -------
        str
            The modified text blob with replaced strings.
        """
        for pattern in self._patterns:
            text_blob = pattern.sub(self.replacement_string, text_blob)
        return text_blob
