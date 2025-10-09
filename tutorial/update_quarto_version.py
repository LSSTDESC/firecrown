#!/usr/bin/env python
"""Update version in _quarto.yml from firecrown/version.py.

This script ensures the tutorial version stays synchronized with the
package version defined in firecrown/version.py.
"""

import re
import sys
from pathlib import Path

# Import version from firecrown package
sys.path.insert(0, str(Path(__file__).parent.parent))
from firecrown.version import __version__  # noqa: E402

# Update _quarto.yml
quarto_file = Path(__file__).parent / "_quarto.yml"
content = quarto_file.read_text()

updated = re.sub(
    r'subtitle: "version [^"]*"', f'subtitle: "version {__version__}"', content
)

if updated != content:
    quarto_file.write_text(updated)
    print(f"✅ Updated _quarto.yml to version {__version__}")
else:
    print(f"ℹ️  _quarto.yml already at version {__version__}")
