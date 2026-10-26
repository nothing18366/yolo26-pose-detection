"""Prepare the six landscape demonstrations, posters and Android catalog."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rehab_service.catalog import ACTIONS

# Selected by checking the action titles and transitions in the original compilation.
CLIPS = {"rub": (11.5, 33.5), "fist": (123, 135), "stretch": (34.5, 59),
         "cup": (76.3, 97.5), "abduction": (99.5, 122), "press": (135.2, 147.4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--android-assets", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("找不到横版合集视频")
    output = ROOT / "media"
    output.mkdir(exist_ok=True)
    for key, (start, end) in CLIPS.items():
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(start), "-i", str(args.source), "-t", str(end - start),
                        "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=1280:720",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output / f"{key}.mp4")], check=True)
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", "1", "-i", str(output / f"{key}.mp4"), "-frames:v", "1",
                        "-vf", "scale=640:360", str(output / f"{key}.jpg")], check=True)
        print("已生成", key, flush=True)
    if args.android_assets:
        args.android_assets.mkdir(parents=True, exist_ok=True)
        (args.android_assets / "rehab_actions.json").write_text(
            json.dumps(ACTIONS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
