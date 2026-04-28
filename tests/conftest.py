"""Pytest configuration shared across the test suite."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import src.*` works when pytest is
# invoked via the bare `pytest` entrypoint (where sys.path[0] points to the
# pytest executable directory).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
