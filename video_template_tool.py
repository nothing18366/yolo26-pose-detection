"""Minimal keyboard-assisted phase annotator for the supplied reference videos.

Usage:
  python video_template_tool.py --video PATH --action rub --output video_templates/rub_hand_back.json

Keys: space pause/play, 1/2/3/4 mark a phase, c mark a cycle, q save and quit.
The tool records frame numbers only; measured tolerances remain in the JSON template.
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_no, paused = 0, False
    phases, cycles = [], []
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
        cv2.putText(frame, f"frame={frame_no}  space=play  1-4=phase  c=cycle  q=save",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
        cv2.imshow("video template annotator", frame)
        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif ord("1") <= key <= ord("4"):
            phases.append({"label": chr(key), "frame": frame_no})
        elif key == ord("c"):
            cycles.append(frame_no)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"action": args.action, "source": str(args.video), "video_fps": fps,
            "phase_marks": phases, "cycle_marks": cycles}
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        old.update(data)
        data = old
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
