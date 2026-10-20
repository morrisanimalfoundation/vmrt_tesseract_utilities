"""
Shared CLI utilities for consistent argument parsing across scripts.
"""


def normalize_flag_spelling(argv: list[str]) -> list[str]:
    """Rewrite '--foo_bar' (and '--foo_bar=value') to '--foo-bar' so either spelling works.

    argparse accepts any unambiguous prefix of a flag, so when both '--foo_bar' and
    '--foo-bar' are defined, shortened forms like '--foo' become ambiguous. This
    normalizer rewrites underscore spellings to hyphens before parsing, letting us
    define only the hyphen form while keeping backwards compatibility.
    """
    normalized = []
    for i, arg in enumerate(argv):
        if arg == '--':
            return normalized + argv[i:]
        if arg.startswith('--'):
            name, sep, value = arg.partition('=')
            arg = name.replace('_', '-') + sep + value
        normalized.append(arg)
    return normalized
