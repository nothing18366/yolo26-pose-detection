"""Shared pose/hand inference used by the Qt app and network service."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

from metrics import ExpSmoother, finger_angles, local_frame, palm_center
from tracking import HandTracker
from video_templates import load as load_template

torch.set_num_threads(4)

MODEL_DIR = Path(__file__).parent / "models"
MODEL = str(MODEL_DIR / "yolo26n-pose.pt")
HAND_MODEL = str(MODEL_DIR / "hand_landmarker.task")
HAND_LINKS = vision.HandLandmarksConnections.HAND_CONNECTIONS
HAND_LINE, HAND_DOT = (0, 255, 0), (0, 0, 255)
IMGSZ = 320
POSE_EVERY_N_FRAMES = 2
TEMPLATE_NAMES = {
    "rub": "rub_hand_back",
    "fist": "fist_extension",
    "stretch": "finger_stretch",
    "cup": "cup_transfer",
    "abduction": "finger_abduction",
    "press": "finger_press",
}


def detect_hands_by_half(landmarker, img, previous_hands=None, guidance_rois=None, pose=None, wrist_crops=False,
                         required_side=None):
    """Retry only until the action's required hands have been found."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assignment = HandTracker()
    assignment.last = previous_hands or {}

    def complete(hands):
        if required_side is None:
            return len(hands) >= 2
        return required_side in assignment.assign(hands, pose, w)

    def convert(det, x0=0, y0=0, cw=w, ch=h):
        out = []
        for i, hl in enumerate(det.hand_landmarks):
            pts = np.array([[lm.x * cw + x0, lm.y * ch + y0] for lm in hl], dtype=float)
            xyz = np.array([[lm.x * cw + x0, lm.y * ch + y0, lm.z * cw] for lm in hl], dtype=float)
            hand = {"cat": det.handedness[i][0].category_name, "pts": pts, "xyz": xyz}
            world = getattr(det, "hand_world_landmarks", [])
            if i < len(world):
                hand["world"] = np.array([[lm.x, lm.y, lm.z] for lm in world[i]], dtype=float)
            out.append(hand)
        return out

    def detect(crop, x0=0, y0=0):
        ch, cw = crop.shape[:2]
        det = landmarker.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        return convert(det, x0, y0, cw, ch)

    if wrist_crops and pose is not None:
        result = []
        for side, elbow_i, wrist_i in (("left", 7, 9), ("right", 8, 10)):
            if required_side is not None and side != required_side:
                continue
            if pose[wrist_i, 2] < 0.5 or pose[elbow_i, 2] < 0.5:
                continue
            wrist, elbow = pose[wrist_i, :2], pose[elbow_i, :2]
            radius = max(220 * w / 720.0, 0.9 * np.linalg.norm(wrist - elbow))
            x0, y0 = max(0, int(wrist[0] - radius)), max(0, int(wrist[1] - radius))
            x1, y1 = min(w, int(wrist[0] + radius)), min(h, int(wrist[1] + radius))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            candidates = detect(rgb[y0:y1, x0:x1], x0, y0)
            if candidates:
                hand = dict(min(candidates, key=lambda h: np.linalg.norm(h["pts"][0] - wrist)))
                if np.linalg.norm(hand["pts"][0] - wrist) < 0.6 * radius:
                    hand["side"] = side
                    result.append(hand)
        return result

    result = convert(landmarker.detect(mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))))
    duplicate = None
    if len(result) == 2:
        hand_size = max(np.linalg.norm(hand["pts"][12] - hand["pts"][0]) for hand in result)
        overlap = np.mean(np.linalg.norm(result[0]["pts"] - result[1]["pts"], axis=1))
        if overlap < 0.25 * hand_size:
            duplicate = result.pop()
    if complete(result):
        return result

    regions = []

    def add_region(region):
        xa, ya, xb, yb = region
        if xb - xa <= 32 or yb - ya <= 32:
            return
        candidate = (max(0, xa), max(0, ya), min(w, xb), min(h, yb))
        if not any(
            max(0, min(candidate[2], old[2]) - max(candidate[0], old[0]))
            * max(0, min(candidate[3], old[3]) - max(candidate[1], old[1]))
            >= 0.7 * min(
                (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
                (old[2] - old[0]) * (old[3] - old[1]),
            )
            for old in regions
        ):
            regions.append(candidate)

    if pose is not None and len(pose) > 10:
        for side, elbow_i, wrist_i in (("right", 8, 10), ("left", 7, 9)):
            if required_side is not None and side != required_side:
                continue
            if pose[elbow_i, 2] < 0.4 or pose[wrist_i, 2] < 0.4:
                continue
            elbow, wrist = pose[elbow_i, :2], pose[wrist_i, :2]
            radius = max(80, 0.55 * np.linalg.norm(elbow - wrist))
            add_region((int(elbow[0] - radius), int(elbow[1] - radius),
                        int(elbow[0] + radius), int(elbow[1] + radius)))
    for side, old in (previous_hands or {}).items():
        if required_side is not None and side != required_side:
            continue
        pts = np.asarray(old.get("pts"), dtype=float)
        if pts.size == 0:
            continue
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)
        pad_x, pad_y = max((x1 - x0) * 0.20, 32), max((y1 - y0) * 0.20, 32)
        add_region((int(x0 - pad_x), int(y0 - pad_y), int(x1 + pad_x), int(y1 + pad_y)))
    for roi in (guidance_rois or {}).values():
        if len(roi) == 4:
            add_region(tuple(int(v) for v in (roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h)))
    if not regions:
        regions = [(0, 0, w // 2, h), (w // 2, 0, w, h)]

    for x0, y0, x1, y1 in regions:
        for hand in detect(rgb[y0:y1, x0:x1], x0, y0):
            if all(np.linalg.norm(hand["pts"][0] - old["pts"][0]) > 35 for old in result):
                result.append(hand)
                if complete(result):
                    break
        if complete(result):
            break
    if duplicate is not None and len(result) == 1:
        result.append(duplicate)
    return result


class PoseEngine:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(MODEL)
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL, delegate=BaseOptions.Delegate.CPU),
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        self.action_key = None
        self.affected_side = None
        self.guidance_rois = {}
        self.reset()

    def set_action_key(self, key, affected_side=None):
        self.action_key = key
        self.affected_side = affected_side
        name = TEMPLATE_NAMES.get(key)
        self.guidance_rois = load_template(name).get("guidance_rois", {}) if name else {}

    def reset(self):
        self.smooth = {"left": ExpSmoother(), "right": ExpSmoother()}
        self.tracker = HandTracker(max_missing_s=0.35)
        self.frame_no = 0
        self.pose = None
        self.pose_result = None

    def process(self, img, timestamp=None):
        started = time.monotonic()
        t = started if timestamp is None else float(timestamp)
        self.frame_no += 1
        if self.pose_result is None or self.frame_no % POSE_EVERY_N_FRAMES == 1:
            result = self.model.predict(
                img, imgsz=IMGSZ, conf=0.5, verbose=False, device=self.device)[0]
            self.pose_result = result
            self.pose = result.keypoints.data[0].cpu().numpy() if len(result.keypoints.data) else None

        h, w = img.shape[:2]
        hlist = detect_hands_by_half(
            self.hands, img, self.tracker.last, self.guidance_rois, self.pose,
            wrist_crops=getattr(self, "action_key", None) == "abduction",
            required_side=(getattr(self, "affected_side", None)
                           if getattr(self, "action_key", None) in ("cup", "press", "abduction") else None))
        hlist = self.tracker.update(hlist, t, self.pose, w)
        for hand in hlist:
            if hand.get("state", "fresh") == "fresh":
                hand["pts"] = self.smooth[hand["side"]].update(hand["pts"])
            hand["angles"] = finger_angles(hand["pts"])
            hand["palm"] = palm_center(hand["pts"])
            hand["wrist"] = hand["pts"][0].copy()

        canvas = self.pose_result.plot(img=img.copy())
        for hand in hlist:
            if hand.get("state", "fresh") != "fresh":
                continue
            for link in HAND_LINKS:
                cv2.line(canvas, tuple(hand["pts"][link.start].astype(int)),
                         tuple(hand["pts"][link.end].astype(int)), HAND_LINE, 2)
            for point in hand["pts"].astype(int):
                cv2.circle(canvas, tuple(point), 3, HAND_DOT, -1)

        axis = None
        if self.pose is not None and len(self.pose) > 10:
            for _side, elbow_i, wrist_i in (("left", 7, 9), ("right", 8, 10)):
                if self.pose[elbow_i, 2] >= .3 and self.pose[wrist_i, 2] >= .3:
                    axis, _ = local_frame(
                        self.pose[elbow_i, :2], self.pose[wrist_i, :2])[:2]
                    break
        hand_tracking = {
            hand["side"]: {
                "state": hand.get("state", "fresh"),
                "age": hand.get("age", 0.0),
                "predicted": bool(hand.get("predicted")),
            }
            for hand in hlist
        }
        frame = {
            "t": t,
            "pose": self.pose,
            "hands": hlist,
            "frame_size": (w, h),
            "forearm_axis": axis,
            "hand_tracking": hand_tracking,
            "tracking_confidence": sum(hand.get("state", "fresh") == "fresh" for hand in hlist) / 2.0,
            "processing_ms": (time.monotonic() - started) * 1000.0,
        }
        return canvas, frame

    def close(self):
        self.hands.close()
