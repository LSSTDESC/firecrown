"""Tests for firecrown.fctools utility modules."""

from rich.console import Console


def strip_rich_markup(text: str) -> str:
    """Return text with all rich color/formatting removed."""
    console = Console(color_system=None, no_color=True, record=True)
    console.print(text)
    return console.export_text()
