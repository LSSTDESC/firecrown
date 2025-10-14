"""Unit tests for firecrown.fctools.list_tools module.

Tests the tool discovery and listing functionality.
"""

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from firecrown.fctools.list_tools import (
    _discover_tools,
    _extract_description_from_docstring,
    _extract_description_from_file,
    app,
)

from . import match_wrapped


class TestExtractDescriptionFromDocstring:
    """Tests for _extract_description_from_docstring function."""

    def test_extract_from_simple_docstring(self):
        """Test extracting description from a simple docstring."""
        docstring = "This is a simple tool."

        result = _extract_description_from_docstring(docstring)

        assert result == "This is a simple tool."

    def test_extract_from_multiline_docstring(self):
        """Test extracting first line from multiline docstring."""
        docstring = """This is the first line.

        This is the second line with more details.
        """

        result = _extract_description_from_docstring(docstring)

        assert result == "This is the first line."

    def test_extract_with_triple_quotes(self):
        """Test handling docstring with triple quotes in text.

        The function filters out lines that start with triple quotes,
        so a docstring that's just the quotes returns the fallback.
        """
        docstring = '"""This is a tool description."""'

        result = _extract_description_from_docstring(docstring)

        # Lines starting with """ are filtered out
        assert result == "Tool description not available"

    def test_extract_from_empty_docstring(self):
        """Test handling empty docstring."""
        docstring = ""

        result = _extract_description_from_docstring(docstring)

        assert result == "Tool description not available"

    def test_extract_from_whitespace_only(self):
        """Test handling whitespace-only docstring."""
        docstring = "   \n   \n   "

        result = _extract_description_from_docstring(docstring)

        assert result == "Tool description not available"

    def test_extract_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        docstring = """

        This is the description.
        """

        result = _extract_description_from_docstring(docstring)

        assert result == "This is the description."

    def test_extract_with_single_quotes(self):
        """Test handling docstring with single triple quotes in text.

        The function filters out lines that start with triple quotes,
        so a docstring that's just the quotes returns the fallback.
        """
        docstring = "'''This tool does something.'''"

        result = _extract_description_from_docstring(docstring)

        # Lines starting with ''' are filtered out
        assert result == "Tool description not available"


class TestExtractDescriptionFromFile:  # pylint: disable=import-outside-toplevel
    """Tests for _extract_description_from_file function."""

    def test_extract_from_valid_python_file(self, tmp_path):
        """Test extracting description from a valid Python file."""
        test_file = tmp_path / "test_tool.py"
        test_file.write_text(
            '"""This is a test tool for testing purposes."""\n\ndef main():\n    pass\n'
        )

        result = _extract_description_from_file(test_file)

        assert "test tool" in result.lower()

    def test_extract_from_file_with_multiline_docstring(self, tmp_path):
        """Test extracting from file with multiline docstring."""
        test_file = tmp_path / "tool.py"
        test_file.write_text(
            '"""First line is important.\n\nSecond line has more details.\n"""\n'
        )

        result = _extract_description_from_file(test_file)

        assert "First line is important" in result

    def test_extract_from_file_without_docstring(self, tmp_path):
        """Test handling file without module docstring."""
        test_file = tmp_path / "no_doc.py"
        test_file.write_text("def main():\n    pass\n")

        result = _extract_description_from_file(test_file)

        assert result == "Tool description not available"

    def test_extract_from_nonexistent_file(self, tmp_path):
        """Test handling nonexistent file."""
        fake_file = tmp_path / "nonexistent_tool.py"
        # Don't create the file, just use the path

        result = _extract_description_from_file(fake_file)

        assert result == "Tool description not available"

    def test_extract_from_file_with_syntax_error(self, tmp_path):
        """Test handling file with syntax errors.

        AST parsing will fail on syntax errors, falling back to the
        generic message.
        """
        test_file = tmp_path / "broken.py"
        test_file.write_text('"""Good docstring."""\n\ndef broken(\n')

        result = _extract_description_from_file(test_file)

        # Syntax error in the code causes AST parsing to fail
        assert result == "Tool description not available"

    def test_extract_from_file_with_invalid_unicode(self, tmp_path):
        """Test handling file with encoding errors."""
        test_file = tmp_path / "bad_encoding.py"
        test_file.write_bytes(b"\xff\xfe\x00\x00")

        result = _extract_description_from_file(test_file)

        assert result == "Tool description not available"

    def test_extract_with_long_docstring(self, tmp_path):
        """Test that long docstrings are truncated."""
        long_desc = "A" * 200  # Longer than 80 chars
        test_file = tmp_path / "long_doc.py"
        test_file.write_text(f'"""{long_desc}"""\n')

        result = _extract_description_from_file(test_file)

        # Should be truncated to around 80 chars
        assert len(result) <= 85  # Allow some margin for ellipsis

    def test_extract_fallback_to_import(self, tmp_path, monkeypatch):
        """Test fallback to importlib when AST fails."""
        # Create a file with a valid docstring
        test_file = tmp_path / "import_test.py"
        test_file.write_text('"""Fallback test tool."""\nx = 1\n')

        # Mock get_module_docstring to raise OSError to trigger fallback
        def mock_get_module_docstring(path):
            raise OSError("Simulated error")

        import firecrown.fctools.list_tools as list_tools_module

        monkeypatch.setattr(
            list_tools_module, "get_module_docstring", mock_get_module_docstring
        )

        result = _extract_description_from_file(test_file)

        # Should get description via importlib fallback
        assert "Fallback test tool" in result

    def test_extract_fallback_with_no_spec_loader(self, tmp_path, monkeypatch):
        """Test fallback when spec.loader is None."""
        test_file = tmp_path / "no_loader.py"
        test_file.write_text('"""Test."""\n')

        # Mock to return a spec without loader
        # pylint: disable=unused-argument
        def mock_spec_from_file_location(name, location):
            class FakeSpec:
                """Mock spec without loader."""

                loader = None

            return FakeSpec()

        import importlib.util

        monkeypatch.setattr(
            importlib.util, "spec_from_file_location", mock_spec_from_file_location
        )

        # Also mock get_module_docstring to fail
        def mock_get_module_docstring(path):
            raise OSError("Simulated error")

        import firecrown.fctools.list_tools as list_tools_module

        monkeypatch.setattr(
            list_tools_module, "get_module_docstring", mock_get_module_docstring
        )

        result = _extract_description_from_file(test_file)

        # Should return fallback since spec.loader is None
        assert result == "Tool description not available"

    def test_extract_fallback_with_empty_docstring(self, tmp_path, monkeypatch):
        """Test fallback when module has no docstring."""
        # Create a file without a docstring
        test_file = tmp_path / "no_doc_fallback.py"
        test_file.write_text("x = 1\n")

        # Mock get_module_docstring to raise OSError to trigger fallback
        def mock_get_module_docstring(path):
            raise OSError("Simulated error")

        import firecrown.fctools.list_tools as list_tools_module

        monkeypatch.setattr(
            list_tools_module, "get_module_docstring", mock_get_module_docstring
        )

        result = _extract_description_from_file(test_file)

        # Should return fallback since module has no __doc__
        assert result == "Tool description not available"


class TestDiscoverTools:  # pylint: disable=import-outside-toplevel
    """Tests for _discover_tools function."""

    def test_discover_finds_tools(self):
        """Test that tool discovery finds real fctools."""
        tools = _discover_tools()

        # Should find at least a few tools
        assert len(tools) > 0
        # Should be a dict with filename -> description mapping
        assert isinstance(tools, dict)
        assert all(isinstance(k, str) for k in tools)
        assert all(isinstance(v, str) for v in tools.values())

    def test_discover_excludes_init(self):
        """Test that __init__.py is excluded from tools."""
        tools = _discover_tools()

        assert "__init__.py" not in tools

    def test_discover_excludes_list_tools(self):
        """Test that list_tools.py excludes itself."""
        tools = _discover_tools()

        assert "list_tools.py" not in tools

    def test_discover_excludes_private_files(self, tmp_path, monkeypatch):
        """Test that files starting with _ are excluded."""
        # First check the real discover_tools
        tools = _discover_tools()
        assert all(not name.startswith("_") for name in tools)

        # Now test with a mock directory that has a private file
        test_dir = tmp_path / "mock_fctools"
        test_dir.mkdir()

        # Create a private file
        private_file = test_dir / "_private_tool.py"
        private_file.write_text('"""Private tool."""\n')

        # Create a public file
        public_file = test_dir / "public_tool.py"
        public_file.write_text('"""Public tool."""\n')

        # Mock __file__ to point to our test directory
        import firecrown.fctools.list_tools as list_tools_module

        original_file = list_tools_module.__file__
        try:
            # Set __file__ to be in our test directory
            monkeypatch.setattr(
                list_tools_module, "__file__", str(test_dir / "list_tools.py")
            )

            tools = _discover_tools()

            # Should include public but not private
            assert "public_tool.py" in tools
            assert "_private_tool.py" not in tools
        finally:
            # Restore original
            monkeypatch.setattr(list_tools_module, "__file__", original_file)

    def test_discover_returns_descriptions(self):
        """Test that discovered tools have descriptions."""
        tools = _discover_tools()

        # All tools should have non-empty descriptions
        for description in tools.values():
            assert description
            assert len(description) > 0

    def test_discover_finds_specific_tools(self):
        """Test that known tools are discovered."""
        tools = _discover_tools()

        # These tools should exist in the fctools directory
        expected_tools = ["common.py", "ast_utils.py"]

        for expected in expected_tools:
            assert expected in tools, f"Expected to find {expected} in discovered tools"


class TestMainFunction:  # pylint: disable=import-outside-toplevel
    """Tests for main CLI function."""

    def test_main_basic_output(self):
        """Test main function produces output."""
        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert match_wrapped(result.stdout, "Available fctools:")

    def test_main_lists_tools(self):
        """Test that main lists actual tools."""
        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        # Should list at least some known tools
        assert match_wrapped(result.stdout, "common.py")
        assert match_wrapped(result.stdout, "ast_utils.py")

    def test_main_verbose_flag(self):
        """Test main with --verbose flag."""
        runner = CliRunner()
        result = runner.invoke(app, ["--verbose"])

        assert result.exit_code == 0
        assert match_wrapped(result.stdout, "Available fctools:")
        assert match_wrapped(result.stdout, "Usage: python -m firecrown.fctools.")

    def test_main_short_verbose_flag(self):
        """Test main with -v flag."""
        runner = CliRunner()
        result = runner.invoke(app, ["-v"])

        assert result.exit_code == 0
        assert match_wrapped(result.stdout, "Usage: python -m firecrown.fctools.")

    def test_main_non_verbose_has_help_text(self):
        """Test that non-verbose mode shows help text."""
        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert match_wrapped(result.stdout, "Use --verbose for detailed information")
        assert match_wrapped(
            result.stdout, "Use 'python -m firecrown.fctools.TOOL --help'"
        )

    def test_main_verbose_shows_tool_details(self):
        """Test that verbose mode shows more details."""
        runner = CliRunner()
        result = runner.invoke(app, ["--verbose"])

        assert result.exit_code == 0
        # In verbose mode, should show module names without .py
        # and usage instructions
        lines = result.stdout.split("\n")
        has_usage_line = any("Usage: python -m" in line for line in lines)
        assert has_usage_line

    def test_main_tools_are_sorted(self):
        """Test that tools are listed in alphabetical order."""
        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0

        # Extract tool names from output
        lines = result.stdout.split("\n")
        tool_lines = [
            line.strip() for line in lines if line.strip() and line.startswith("  ")
        ]

        # Get tool names (first word after whitespace)
        tool_names = []
        for line in tool_lines:
            parts = line.split()
            if parts and parts[0].endswith(".py"):
                tool_names.append(parts[0])

        # Check they're sorted
        if len(tool_names) > 1:
            assert tool_names == sorted(tool_names)

    def test_main_with_subprocess(self):
        """Test that the script can be executed directly via subprocess.

        This test verifies that the __main__ block and ImportError fallback
        work correctly when the script is run directly.
        """
        # Execute the script directly to test __main__ block
        script_path = "firecrown/fctools/list_tools.py"
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert match_wrapped(result.stdout, "Available fctools:")
        assert "common.py" in result.stdout

    def test_main_subprocess_with_verbose(self):
        """Test script execution with verbose flag via subprocess."""
        script_path = "firecrown/fctools/list_tools.py"
        result = subprocess.run(
            [sys.executable, script_path, "--verbose"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert match_wrapped(result.stdout, "Available fctools:")
        assert match_wrapped(result.stdout, "Usage: python -m firecrown.fctools.")


class TestIntegration:  # pylint: disable=import-outside-toplevel
    """Integration tests for list_tools functionality."""

    def test_full_workflow(self):
        """Test the complete workflow of discovering and listing tools."""
        # Discover tools
        tools = _discover_tools()

        assert len(tools) > 0

        # Verify each tool file exists
        from firecrown.fctools import list_tools

        fctools_dir = Path(list_tools.__file__).parent

        for tool_name in tools:
            tool_path = fctools_dir / tool_name
            assert tool_path.exists(), f"Tool file {tool_name} should exist"
            assert (
                tool_path.suffix == ".py"
            ), f"Tool {tool_name} should be a Python file"

    def test_all_tools_have_valid_descriptions(self):
        """Test that all discovered tools have meaningful descriptions."""
        tools = _discover_tools()

        for description in tools.values():
            # Description should not be the fallback message for real tools
            # (though some tools might legitimately not have docstrings)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_cli_output_matches_discovery(self):
        """Test that CLI output matches tool discovery results."""
        tools = _discover_tools()

        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0

        # Every discovered tool should appear in the output
        for tool_name in tools:
            assert tool_name in result.stdout, f"Tool {tool_name} should be in output"
