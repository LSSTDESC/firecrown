"""Tests for firecrown.fctools utility modules."""

import re

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_rich_markup(text: str) -> str:
    """Remove ANSI codes and non-ASCII characters."""
    text = _ANSI_RE.sub("", text)
    return text.encode("ascii", "ignore").decode("ascii")


def match_wrapped(text: str, phrase: str) -> bool:
    """
    Return True if all words in `phrase` appear in `text` in order,
    possibly separated by whitespace, newlines, or ANSI codes.
    """
    # Escape regex metacharacters in the phrase, split into words
    clean_text = strip_rich_markup(text)
    words = [re.escape(w) for w in phrase.split()]
    # Join with a pattern that allows arbitrary whitespace (including newlines)
    pattern = r"\s+".join(words)
    return re.search(pattern, clean_text, flags=re.MULTILINE) is not None
