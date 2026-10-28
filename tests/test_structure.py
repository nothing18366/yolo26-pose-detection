"""Regression checks for source relocation, resources and algorithm identity."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehab_service.api import rtc
from rehab_service.recognition.actions import REGISTRY
from rehab_service.recognition.inference import PoseEngine, TEMPLATE_NAMES
from rehab_service.recognition.templates import load
from rehab_service.settings import MEDIA_DIR, MODEL_DIR, PROJECT_ROOT, TEMPLATE_DIR
from rehab_service.storage import database


class StructureTests(unittest.TestCase):
    def test_shared_engine_and_six_action_templates(self):
        self.assertIs(rtc.PoseEngine, PoseEngine)
        self.assertEqual([action.key for action in REGISTRY],
                         ["rub", "fist", "stretch", "cup", "abduction", "press"])
        for action in REGISTRY:
            self.assertTrue(load(TEMPLATE_NAMES[action.key]))

    def test_assets_resolve_independently_of_working_directory(self):
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        self.assertTrue((MODEL_DIR / "yolo26n-pose.pt").is_file())
        self.assertTrue((MODEL_DIR / "hand_landmarker.task").is_file())
        self.assertEqual(len(list(TEMPLATE_DIR.glob("*.json"))), 6)
        for action in REGISTRY:
            for extension in ("mp4", "jpg", "srt"):
                self.assertTrue((MEDIA_DIR / f"{action.key}.{extension}").is_file())

    def test_layout_only_change_preserves_trend_identity(self):
        self.assertEqual(database.RehabStore.algorithm_fingerprint(), "989357def6b46bd9")
        with tempfile.TemporaryDirectory() as directory, patch.object(database, "MODEL_DIR", Path(directory)):
            self.assertNotEqual(database.RehabStore.algorithm_fingerprint(), "989357def6b46bd9")


if __name__ == "__main__":
    unittest.main()
