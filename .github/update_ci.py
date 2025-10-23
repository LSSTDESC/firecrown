"""Update the CI environment file."""

import sys

python_version = sys.argv[1]

with open("environment.yml", encoding="utf-8") as fh:
    lines = fh.readlines()
found = False
for i, line in enumerate(lines):
    if line.startswith("  - python"):
        lines[i] = f"  - python={python_version}\n"
        found = True
        break

if found:
    with open("env_tmp.yml", "w", encoding="utf-8") as f:
        f.writelines(lines)
    sys.exit(0)

sys.exit(1)
