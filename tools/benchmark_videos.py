"""Repeatable six-video input at 640x480; run before and after display changes."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rehab_service.recognition.inference import PoseEngine
from rehab_service.services.training import TrainingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-draw", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    engine = PoseEngine()
    try:
        for key in ("rub", "fist", "stretch", "cup", "abduction", "press"):
            source = ROOT / "media" / (key + ".mp4")
            capture = cv2.VideoCapture(str(source))
            assert capture.isOpened(), source
            fps = capture.get(cv2.CAP_PROP_FPS)
            assert fps > 0
            engine.reset()
            engine.set_action_key(key, "right")
            session = TrainingSession({"action": key, "patient": {"name": "固定视频", "affected_side": "right", "forearm_cm": 25, "target_reps": 99}})
            index = 0
            started = time.monotonic()
            with (args.output / (key + ".jsonl")).open("x") as output:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    height, width = image.shape[:2]
                    scale = min(640 / width, 480 / height)
                    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    y, x = (480 - resized.shape[0]) // 2, (640 - resized.shape[1]) // 2
                    frame[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                    kwargs = {"draw": False} if args.no_draw else {}
                    _, metrics = engine.process(frame, index / fps, **kwargs)
                    result = session.update(metrics)
                    # Wall-clock measurements cannot participate in an algorithm equality check.
                    result.pop("processing_ms", None)
                    output.write(json.dumps({"index": index, "input_sha256": hashlib.sha256(frame).hexdigest(),
                        "t": metrics["t"], "pose": metrics["pose"], "hands": metrics["hands"],
                        "frame_size": metrics["frame_size"], "result": result,
                        "processing_ms": metrics["processing_ms"], "drawing_ms": metrics.get("drawing_ms")},
                        ensure_ascii=False, default=lambda value: value.tolist()) + "\n")
                    index += 1
            capture.release()
            summary = dict(session.summary(), frames=index, input_fps=fps, elapsed=time.monotonic() - started,
                           video_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            (args.output / (key + "-summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            print(key, index, "frames", summary["count"], "reps", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
