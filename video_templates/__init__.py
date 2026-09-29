"""Parameters measured from the supplied standard demonstration videos."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def load(name):
    with (ROOT / (name + ".json")).open(encoding="utf-8") as f:
        return json.load(f)
