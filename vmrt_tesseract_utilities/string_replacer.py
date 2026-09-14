import re
from typing import List

"""
Provides string replacement functionality.
"""

# Matches GRLS-style subject IDs (a numeric site prefix, a dash, and a numeric
# suffix, e.g. "094-000520") so that variant formats appearing in OCR'd text can
# also be matched.
GRLS_ID_RE = re.compile(r'^(\d+)-(\d+)$')


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
        self.target_strings = target_strings
        self.replacement_string = replacement_string
        self._patterns = [self._build_pattern(target) for target in target_strings]

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
        match = GRLS_ID_RE.match(target)
        if not match:
            return [target]
        prefix, suffix = match.groups()
        padded_suffix = f'0{suffix}'
        variants = [
            target,
            f'{prefix}{suffix}',
            f'{prefix}-{padded_suffix}',
            f'{prefix}{padded_suffix}',
        ]
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(variants))

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
        # Longest first, so a variant that's a prefix of another isn't matched short.
        variants = sorted(cls._expand_id_variants(target), key=len, reverse=True)
        alternation = '|'.join(re.escape(variant) for variant in variants)
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
