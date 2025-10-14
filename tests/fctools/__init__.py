"""Tests for firecrown.fctools utility modules."""

import re

# Regular expression to match ANSI escape sequences.
# ANSI sequences start with ESC (\x1B) followed by either:
# - A single character in the range [@-Z\\-_] (Fe sequences)
# - Or a Control Sequence Introducer (CSI): [ followed by parameter bytes
#   [0-?]*, intermediate bytes [ -/]*, and a final byte [@-~]
# This pattern matches standard terminal color codes, cursor movement, and
# other formatting sequences commonly used by tools like Rich and colorama.
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_rich_markup(text: str) -> str:
    """Remove ANSI codes and non-ASCII characters."""
    text = _ANSI_RE.sub("", text)
    return text.encode("ascii", "ignore").decode("ascii")


def match_wrapped(text: str, phrase: str) -> bool:
    """
    Return True if the phrase appears in text, allowing for line wrapping.

    Line wrapping means that any position in the phrase may be followed by
    whitespace (including newlines) in the text. This handles cases where
    words are broken across lines.
    """
    clean_text = strip_rich_markup(text)
    # Build a pattern that allows optional whitespace after each character
    pattern_parts = []
    for char in phrase:
        if char.isspace():
            # Spaces in the phrase should match one or more whitespace chars
            pattern_parts.append(r"\s+")
        else:
            # Non-space characters: escape and allow optional whitespace after
            pattern_parts.append(re.escape(char) + r"\s*")
    # Remove trailing \s* from the last character
    pattern = "".join(pattern_parts).rstrip(r"\s*")
    return re.search(pattern, clean_text, flags=re.MULTILINE) is not None
