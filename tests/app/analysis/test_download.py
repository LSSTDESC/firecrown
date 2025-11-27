"""Unit tests for firecrown.app.analysis download utilities.

Tests download and file utility functions.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from rich.console import Console

from firecrown.app.analysis import download_from_url, copy_template


class TestDownloadFromUrl:
    """Tests for download_from_url function."""

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_success(self, mock_retrieve, tmp_path: Path) -> None:
        """Test successful file download."""
        output_file = tmp_path / "test.txt"
        console = Console()

        download_from_url("http://example.com/test.txt", output_file, console)

        mock_retrieve.assert_called_once()

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_with_description(
        self, mock_retrieve, tmp_path: Path
    ) -> None:
        """Test download with custom description."""
        output_file = tmp_path / "test.txt"
        console = Console()

        download_from_url(
            "http://example.com/test.txt",
            output_file,
            console,
            description="Custom download",
        )

        mock_retrieve.assert_called_once()

    @patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
    def test_download_from_url_network_error(self, tmp_path: Path) -> None:
        """Test download with network error."""
        output_file = tmp_path / "test.txt"
        console = Console()

        with pytest.raises(Exception, match="Network error"):
            download_from_url("http://example.com/test.txt", output_file, console)

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_console_parameter(
        self, _mock_retrieve, tmp_path: Path
    ) -> None:
        """Test that console parameter is required."""
        output_file = tmp_path / "test.txt"
        console = Console()

        # Should not raise error when console is provided
        download_from_url("http://example.com/test.txt", output_file, console)


class TestCopyTemplate:
    """Tests for copy_template function."""

    def test_copy_template_success(self, tmp_path: Path) -> None:
        """Test successful template copy."""
        # Create a mock module with a __file__ attribute
        mock_module = MagicMock()
        template_file = tmp_path / "template.py"
        template_file.write_text("# template code")
        mock_module.__file__ = str(template_file)

        dest_file = tmp_path / "output.py"

        copy_template(mock_module, dest_file)

        assert dest_file.exists()
        assert dest_file.read_text() == "# template code"

    def test_copy_template_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test that copy_template creates parent directories."""
        mock_module = MagicMock()
        template_file = tmp_path / "template.py"
        template_file.write_text("# template code")
        mock_module.__file__ = str(template_file)

        dest_file = tmp_path / "subdir" / "output.py"
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        copy_template(mock_module, dest_file)

        assert dest_file.parent.exists()
        assert dest_file.exists()

    def test_copy_template_preserves_content(self, tmp_path: Path) -> None:
        """Test that copy_template preserves file content."""
        mock_module = MagicMock()
        template_file = tmp_path / "template.yaml"
        template_content = "key: value\nnumber: 42"
        template_file.write_text(template_content)
        mock_module.__file__ = str(template_file)

        dest_file = tmp_path / "output.yaml"

        copy_template(mock_module, dest_file)

        assert dest_file.read_text() == template_content

    def test_copy_template_nonexistent_source(self, tmp_path: Path) -> None:
        """Test copy_template with nonexistent source file."""
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "nonexistent.py")

        dest_file = tmp_path / "output.py"

        with pytest.raises(FileNotFoundError):
            copy_template(mock_module, dest_file)

    def test_copy_template_directory_destination(self, tmp_path: Path) -> None:
        """Test copy_template with directory as destination."""
        mock_module = MagicMock()
        template_file = tmp_path / "template.py"
        template_file.write_text("# template code")
        mock_module.__file__ = str(template_file)

        dest_dir = tmp_path / "output_dir"
        dest_dir.mkdir()
        dest_file = dest_dir / "template.py"

        copy_template(mock_module, dest_file)

        assert dest_file.exists()
