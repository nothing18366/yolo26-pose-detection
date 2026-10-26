"""Parameters measured from the supplied standard demonstration videos."""

import json

from rehab_service.settings import TEMPLATE_DIR

ROOT = TEMPLATE_DIR


def load(name):
    with (ROOT / (name + ".json")).open(encoding="utf-8") as f:
        return json.load(f)
