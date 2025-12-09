"""Unit tests for firecrown.app.analysis download utilities.

Tests download and file utility functions.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from rich.console import Console

from firecrown.app.analysis import download_from_url, copy_template
from firecrown.app.analysis._download import get_cache_dir, get_cached_filename


class TestCacheHelpers:
    """Tests for cache helper functions."""

    def test_get_cache_dir_creates_directory(self) -> None:
        """Test that get_cache_dir creates the cache directory."""
        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir == Path.home() / ".firecrown" / "sacc_files"

    def test_get_cached_filename_from_url(self) -> None:
        """Test extracting filename from URL."""
        url = "https://example.com/path/to/file.fits"
        filename = get_cached_filename(url)
        assert filename == "file.fits"

    def test_get_cached_filename_with_trailing_slash(self) -> None:
        """Test extracting filename from URL with trailing slash."""
        url = "https://example.com/path/to/file.fits/"
        filename = get_cached_filename(url)
        assert filename == "file.fits"

    def test_get_cached_filename_rejects_bare_domain(self) -> None:
        """Test that bare domain URLs are rejected."""
        with pytest.raises(AssertionError, match="bare domain without a filename"):
            get_cached_filename("https://example.com")

    def test_get_cached_filename_rejects_path_only(self) -> None:
        """Test that URLs ending with path separator are rejected."""
        with pytest.raises(AssertionError, match="lack an extension"):
            get_cached_filename("https://example.com/path/")

    def test_get_cached_filename_rejects_no_extension(self) -> None:
        """Test that filenames without extensions are rejected."""
        with pytest.raises(AssertionError, match="lack an extension"):
            get_cached_filename("https://example.com/path/to/endpoint")

    def test_get_cached_filename_rejects_malformed_url(self) -> None:
        """Test that malformed URLs are rejected."""
        with pytest.raises(AssertionError, match="Invalid filename extracted"):
            # This would extract "https:" as filename if not validated
            get_cached_filename("https:")

    def test_get_cached_filename_accepts_various_extensions(self) -> None:
        """Test that various file extensions are accepted."""
        test_cases = [
            ("https://example.com/data.fits", "data.fits"),
            ("https://example.com/archive.tar.gz", "archive.tar.gz"),
            ("https://example.com/config.yaml", "config.yaml"),
            ("https://example.com/output.hdf5", "output.hdf5"),
        ]
        for url, expected in test_cases:
            assert get_cached_filename(url) == expected


class TestDownloadFromUrl:
    """Tests for download_from_url function."""

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_uses_cache_when_available(
        self, mock_retrieve, tmp_path: Path
    ) -> None:
        """Test that cached files are used instead of downloading."""
        output_file = tmp_path / "test.txt"
        console = Console()

        # Create a cached file
        cache_dir = get_cache_dir()
        cached_file = cache_dir / "test.txt"
        cached_file.write_text("cached content")

        try:
            download_from_url("http://example.com/test.txt", output_file, console)

            # Should not download when cache exists
            mock_retrieve.assert_not_called()

            # Should copy from cache
            assert output_file.exists()
            assert output_file.read_text() == "cached content"
        finally:
            # Cleanup cache
            if cached_file.exists():
                cached_file.unlink()

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_downloads_when_not_cached(
        self, mock_retrieve, tmp_path: Path
    ) -> None:
        """Test that file is downloaded when not in cache."""
        output_file = tmp_path / "test_new.txt"
        console = Console()

        # Remove cached file if it exists
        cache_dir = get_cache_dir()
        cached_file = cache_dir / "test_new.txt"
        if cached_file.exists():
            cached_file.unlink()

        try:
            # Mock the download to create the cached file
            def mock_download(url, path):
                assert url is not None
                Path(path).write_text("downloaded content", encoding="utf-8")

            mock_retrieve.side_effect = mock_download

            download_from_url("http://example.com/test_new.txt", output_file, console)

            # Should download when cache doesn't exist
            mock_retrieve.assert_called_once()

            # Should save to both cache and output
            assert cached_file.exists()
            assert output_file.exists()
            assert output_file.read_text() == "downloaded content"
        finally:
            # Cleanup cache
            if cached_file.exists():
                cached_file.unlink()

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_with_description(
        self, mock_retrieve, tmp_path: Path
    ) -> None:
        """Test download with custom description."""
        output_file = tmp_path / "test_desc.txt"
        console = Console()

        # Remove cached file if it exists
        cache_dir = get_cache_dir()
        cached_file = cache_dir / "test_desc.txt"
        if cached_file.exists():
            cached_file.unlink()

        try:
            # Mock the download
            def mock_download(url, path):
                assert url is not None
                Path(path).write_text("content", encoding="utf-8")

            mock_retrieve.side_effect = mock_download

            download_from_url(
                "http://example.com/test_desc.txt",
                output_file,
                console,
                description="Custom download",
            )

            mock_retrieve.assert_called_once()
        finally:
            # Cleanup cache
            if cached_file.exists():
                cached_file.unlink()

    @patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
    def test_download_from_url_network_error(self, tmp_path: Path) -> None:
        """Test download with network error when file not cached."""
        output_file = tmp_path / "test_error.txt"
        console = Console()

        # Remove cached file if it exists
        cache_dir = get_cache_dir()
        cached_file = cache_dir / "test_error.txt"
        if cached_file.exists():
            cached_file.unlink()

        try:
            with pytest.raises(SystemExit) as exc_info:
                download_from_url(
                    "http://example.com/test_error.txt", output_file, console
                )
            assert exc_info.value.code == 1
        finally:
            # Cleanup cache
            if cached_file.exists():
                cached_file.unlink()

    @patch("urllib.request.urlretrieve")
    def test_download_from_url_console_parameter(
        self, mock_retrieve, tmp_path: Path
    ) -> None:
        """Test that console parameter is required."""
        output_file = tmp_path / "test_console.txt"
        console = Console()

        # Remove cached file if it exists
        cache_dir = get_cache_dir()
        cached_file = cache_dir / "test_console.txt"
        if cached_file.exists():
            cached_file.unlink()

        try:
            # Mock the download
            def mock_download(_, path):
                Path(path).write_text("content", encoding="utf-8")

            mock_retrieve.side_effect = mock_download

            # Should not raise error when console is provided
            download_from_url(
                "http://example.com/test_console.txt", output_file, console
            )
        finally:
            # Cleanup cache
            if cached_file.exists():
                cached_file.unlink()


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
