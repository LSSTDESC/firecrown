#!/usr/bin/env python3
"""Exit 0 if $REF_NAME is in the supported CI branch list, else exit 1."""

import json
import os
import sys

branch = os.environ["REF_NAME"]
with open(".github/ci-branches.json") as f:
    branches = json.load(f)
sys.exit(0 if branch in branches else 1)
