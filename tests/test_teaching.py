"""Check the deployed teaching bundle, including SRT times against actual clips."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.storage.database import RehabStore
from rehab_service.services.teaching import TeachingService


class TeachingBundleTests(unittest.TestCase):
    def test_six_complete_pending_bundles_and_subtitle_timelines(self):
        with tempfile.TemporaryDirectory() as directory:
            teaching = TeachingService(RehabStore(Path(directory) / "db.sqlite3"))
            self.assertEqual(len(teaching.items), 6)
            for item in teaching.catalog():
                self.assertEqual(item["review"]["state"], "pending")
                self.assertIsNone(item["review"]["reviewer"])
                self.assertEqual(set(item["files"]), {"video", "poster", "subtitle"})
                for meta in item["files"].values():
                    file = teaching.root / Path(meta["url"]).name
                    self.assertEqual(file.stat().st_size, meta["bytes"])
                    self.assertRegex(meta["sha256"], "^[0-9a-f]{64}$")
                srt = (teaching.root / (item["key"] + ".srt")).read_text(encoding="utf-8")
                def milliseconds(value):
                    h, m, s, ms = map(int, re.split("[:,]", value))
                    return h * 3600000 + m * 60000 + s * 1000 + ms
                cues = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt)
                self.assertEqual(len(cues), 3)
                previous = 0
                for start, end in cues:
                    start, end = milliseconds(start), milliseconds(end)
                    self.assertGreaterEqual(start, previous); self.assertGreater(end, start); previous = end
                if shutil.which("ffprobe"):
                    metadata = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(teaching.root / (item["key"] + ".mp4"))])
                    self.assertLessEqual(previous, float(json.loads(metadata)["format"]["duration"]) * 1000 + 80)


if __name__ == "__main__": unittest.main()
