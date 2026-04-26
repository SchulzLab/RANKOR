#!/usr/bin/env python3
"""
Legacy entrypoint wrapper.

The original repository contained an inference script named `RANKOR_prior.py`.
For the GitHub-friendly version, the maintained CLI is:
  `rankor/cli.py`

This wrapper preserves the legacy filename while delegating to the new CLI.
"""

import os
import sys


def _ensure_local_imports() -> None:
    # Make the folder containing this file importable as a top-level package root.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_ensure_local_imports()

from rankor.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

