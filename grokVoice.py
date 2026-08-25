#!/usr/bin/env python3
"""Launcher for Grapefruit. Prefer: python grokVoice.py  or  python -m grapefruit"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grapefruit.app import main

if __name__ == "__main__":
    main()
