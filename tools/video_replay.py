"""用标准视频离线回放动作状态机。

运行：
  python tools/video_replay.py
  可用 --video PATH --action rub|fist|stretch|cup|abduction|press --side left|right|both。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actions import (ActionContext, CupTransfer, FingerAbduction,
                      FingerPress, FingerStretch, FistOpenClose, RubBackOfHand)
from inference import PoseEngine

VIDEOS = {
    "rub": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/摩擦手背.mp4"),
    "fist": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/握拳伸展.mp4"),
    "stretch": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/牵伸手指.mp4"),
}
CLASSES = {
    "rub": RubBackOfHand, "fist": FistOpenClose, "stretch": FingerStretch,
    "cup": CupTransfer, "abduction": FingerAbduction, "press": FingerPress,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--action", choices=CLASSES, default=None)
    ap.add_argument("--side", choices=("left", "right", "both"), default="left")
    ap.add_argument("--start", type=float, default=0.0, help="视频起始秒数")
    ap.add_argument("--end", type=float, help="视频结束秒数")
    ap.add_argument("--trace", type=Path, help="保存逐帧关键点，供离线核对")
    ap.add_argument("--stride", type=int, default=1, help="离线采样步长，时间戳仍按原视频计算")
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride 必须至少为1")
    jobs = [(args.action, args.video)] if args.video and args.action else list(VIDEOS.items())
    engine = PoseEngine()
    results = []
    trace = []
    for key, path in jobs:
        engine.reset()
        engine.set_action_key(key, args.side if args.side != "both" else None)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if args.video and args.start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
        sessions = []
        for side in (("left", "right") if args.side == "both" else (args.side,)):
            ctx = ActionContext("标准视频", side, 25.0, 99)
            action = CLASSES[key](ctx)
            sessions.append((ctx, action))
        start = time.monotonic(); frames = 0; processed_frames = 0; hand_frames = 0; pose_frames = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            frames += 1
            video_t = args.start + (frames - 1) / fps if args.video else (frames - 1) / fps
            if args.video and args.end is not None and video_t > args.end:
                break
            if (frames - 1) % args.stride:
                continue
            processed_frames += 1
            _, frame = engine.process(img, video_t)
            if frame["pose"] is not None:
                pose_frames += 1
            hand_frames += sum(h.get("state") == "fresh" for h in frame["hands"])
            for ctx, action in sessions:
                ctx.update_forearm(frame["pose"])
                action.update(frame)
            if args.trace:
                trace.append({"t": video_t, "pose": frame["pose"], "hands": [
                    {"side": h["side"], "state": h["state"], "pts": h["pts"],
                     "palm": h["palm"], "wrist": h["wrist"], "world": h.get("world")} for h in frame["hands"]
                ], "frame_size": frame["frame_size"]})
        cap.release()
        for ctx, action in sessions:
            summary = action.summary()
            summary.update({"video": str(path), "frames": frames, "pose_frames": pose_frames,
                            "processed_frames": processed_frames,
                            "side": ctx.affected,
                            "hand_detections": hand_frames, "fps": fps,
                            "start": args.start if args.video else 0.0,
                            "end": args.end if args.video else None,
                            "elapsed": round(time.monotonic() - start, 2),
                            "phase": action.phase})
            results.append(summary)
    engine.close()
    if args.trace:
        args.trace.write_text(json.dumps(trace, default=lambda v: v.tolist()), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
