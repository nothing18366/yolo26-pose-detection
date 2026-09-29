"""用标准视频离线回放动作状态机。

运行：
  /opt/anaconda3/envs/yolo26-pose/bin/python video_replay.py
  可用 --video PATH --action rub|fist|stretch|cup|abduction|press --side left|right。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

from actions import (ActionContext, CupTransfer, FingerAbduction,
                      FingerPress, FingerStretch, FistOpenClose, RubBackOfHand)
from metrics import finger_angles, palm_center
from tracking import HandTracker

ROOT = Path(__file__).parent
MODELS = ROOT / "yolo26n-pose.pt"
HAND_MODEL = ROOT / "hand_landmarker.task"
VIDEOS = {
    "rub": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/摩擦手背.mp4"),
    "fist": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/握拳伸展.mp4"),
    "stretch": Path("/Users/nothing/Desktop/偏瘫手康复操标准动作/牵伸手指.mp4"),
}
CLASSES = {
    "rub": RubBackOfHand, "fist": FistOpenClose, "stretch": FingerStretch,
    "cup": CupTransfer, "abduction": FingerAbduction, "press": FingerPress,
}


def detect_hands(landmarker, img):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def convert(det, x0=0, cw=w):
        out = []
        for i, lm in enumerate(det.hand_landmarks):
            pts = np.array([[p.x * cw + x0, p.y * h] for p in lm], dtype=float)
            xyz = np.array([[p.x * cw + x0, p.y * h, p.z * cw] for p in lm], dtype=float)
            out.append({"cat": det.handedness[i][0].category_name, "pts": pts, "xyz": xyz})
        return out

    result = convert(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))))
    if len(result) >= 2:
        return result
    for x0, x1 in ((0, w // 2), (w // 2, w)):
        crop = np.ascontiguousarray(rgb[:, x0:x1])
        for hand in convert(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)), x0, x1 - x0):
            if all(np.linalg.norm(hand["pts"][0] - old["pts"][0]) > 35 for old in result):
                result.append(hand)
    return result


def sides(hands):
    if len(hands) == 2:
        a, b = sorted(hands, key=lambda x: x["pts"][0, 0])
        a["side"], b["side"] = "right", "left"
    elif hands:
        c = hands[0]["cat"].lower()
        hands[0]["side"] = "right" if c == "left" else "left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--action", choices=CLASSES, default=None)
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--start", type=float, default=0.0, help="视频起始秒数")
    ap.add_argument("--end", type=float, help="视频结束秒数")
    args = ap.parse_args()
    jobs = [(args.action, args.video)] if args.video and args.action else list(VIDEOS.items())
    model = YOLO(str(MODELS))
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL), delegate=BaseOptions.Delegate.CPU),
        num_hands=2, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
    results = []
    for key, path in jobs:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if args.video and args.start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
        ctx = ActionContext("标准视频", args.side, 25.0, 99)
        action = CLASSES[key](ctx) if key != "stretch" else FingerStretch(ctx, "auto")
        tracker = HandTracker(max_missing_s=0.35)
        start = time.monotonic(); frames = 0; hand_frames = 0; pose_frames = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            frames += 1
            video_t = args.start + (frames - 1) / fps if args.video else (frames - 1) / fps
            if args.video and args.end is not None and video_t > args.end:
                break
            h, w = img.shape[:2]
            pose_result = model.predict(img, imgsz=320, conf=0.5, verbose=False)[0]
            pose = pose_result.keypoints.data[0].cpu().numpy() if len(pose_result.keypoints.data) else None
            if pose is not None:
                pose_frames += 1; ctx.update_forearm(pose)
            hs = detect_hands(landmarker, img)
            hs = tracker.update(hs, video_t)
            hand_frames += sum(not h.get("stale") for h in hs)
            for hd in hs:
                hd["angles"] = finger_angles(hd["pts"]); hd["palm"] = palm_center(hd["pts"])
                hd["wrist"] = hd["pts"][0].copy()
            action.update({"t": video_t, "pose": pose, "hands": hs, "frame_size": (w, h)})
        cap.release()
        summary = action.summary()
        summary.update({"video": str(path), "frames": frames, "pose_frames": pose_frames,
                        "hand_detections": hand_frames, "fps": fps,
                        "start": args.start if args.video else 0.0,
                        "end": args.end if args.video else None,
                        "elapsed": round(time.monotonic() - start, 2),
                        "phase": action.phase})
        results.append(summary)
    landmarker.close()
    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
