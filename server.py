"""Command-line entry point; implementation lives in rehab_service.app."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rehab_service.app import app, main

if __name__ == "__main__":
    main()
