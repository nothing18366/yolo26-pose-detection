"""康复动作标准判定：每个动作是一个独立状态机，新增动作 = 加一个子类 + 注册。

阈值来自视频模板（见 video_templates/），当前不使用患者个性化基线。
"""

import numpy as np

from metrics import (
    ELBOW,
    FINGERS,
    FOUR,
    WRIST,
    curl_metric,
    dist,
    finger_extension_features,
    fingers_closed,
    fingers_open,
    fingertips,
    forearm_axis,
    line_angle,
    local_frame,
    roi_contains,
    project_local,
    hand_features,
    isolated_finger_features,
    stretch_features,
    palm_center,
)
from video_templates import load as load_template

# ---- 阈值（动作标准原文） ----
OPEN_DEG = 150.0        # 张开：四指 MCP/PIP 均 ≥150°
CLOSED_DEG = 100.0      # 握拳：四指 MCP/PIP 均 ≤100°
AXIS_MAX_DEG = 30.0     # 运动方向与轴线夹角 ≤30°
DRIFT_MAX_RATIO = 0.2   # 患侧腕漂移 ≤0.2 前臂长
PAUSE_MAX_S = 2.0       # 无 >2s 停顿
MISSING_TOL_S = 0.5

RUB_TEMPLATE = load_template("rub_hand_back")
FIST_TEMPLATE = load_template("fist_extension")
STRETCH_TEMPLATE = load_template("finger_stretch")
CUP_TEMPLATE = load_template("cup_transfer")
ABDUCTION_TEMPLATE = load_template("finger_abduction")
PRESS_TEMPLATE = load_template("finger_press")
RUB_MIN_RATIO = RUB_TEMPLATE["min_span_ratio"]
RUB_MIN_S, RUB_MAX_S = RUB_TEMPLATE["min_duration"], RUB_TEMPLATE["max_duration"]
RUB_CONTACT_RATIO = RUB_TEMPLATE["contact_ratio"]
RUB_KEEP_CONTACT_RATIO = RUB_TEMPLATE.get("keep_contact_ratio", RUB_CONTACT_RATIO * 1.5)
RUB_START_HAND_CONTACT_RATIO = RUB_TEMPLATE["start_hand_contact_ratio"]
RUB_FAR_RATIO = RUB_TEMPLATE.get("far_point_ratio", RUB_MIN_RATIO)
RUB_ORIGIN_TOLERANCE = RUB_TEMPLATE.get("origin_tolerance_ratio", 0.12)
RUB_RETURN_SPAN_TOLERANCE = RUB_TEMPLATE["return_span_tolerance"]
RUB_MAX_PROJECTED_SPAN_RATIO = RUB_TEMPLATE["max_projected_span_ratio"]
RUB_MAX_FOREARM_SCALE_CHANGE = RUB_TEMPLATE["max_forearm_scale_change"]
RUB_MIN_HEALTHY_MOTION = RUB_TEMPLATE["min_healthy_motion_ratio"]
RUB_HEALTHY_MOTION_ADVANTAGE = RUB_TEMPLATE["healthy_motion_advantage"]
RUB_MAX_AFFECTED_MOTION = RUB_TEMPLATE["max_affected_motion_ratio"]
RUB_MAX_HAND_POSE_GAP = RUB_TEMPLATE["max_hand_pose_gap_ratio"]
RUB_START_REGION = RUB_TEMPLATE.get("start_region_ratio", max(0.25, RUB_ORIGIN_TOLERANCE * 2.0))
RUB_MAX_LATERAL_RATIO = RUB_TEMPLATE.get("max_lateral_ratio", 0.65)
RUB_MISSING_TOL_S = RUB_TEMPLATE.get("missing_frame_tolerance", MISSING_TOL_S)
RUB_MIN_MOVE_RATIO = 0.15  # 至少滑动这么多才认为真的做了一次往返（防止静止误计数）

FIST_MIN_S, FIST_MAX_S = FIST_TEMPLATE["min_duration"], FIST_TEMPLATE["max_duration"]

STRETCH_MODE_NAME = {"ext": "背伸", "flex": "掌屈"}
STABLE_FRAMES = 5


# ---- 关键点取值 ----
def get_hand(f, side):
    for h in f.get("hands", []):
        if h["side"] == side:
            return h
    return None


def hand_state(hand):
    return "fresh" if hand is None else hand.get("state", "fresh")


def pose_pt(f, idx, min_conf=0.3):
    pose = f.get("pose")
    if pose is None or len(pose) <= idx:
        return None
    x, y, c = pose[idx][:3]
    if float(c) < min_conf:
        return None
    return np.array([float(x), float(y)])


def elbow_px(f, side):
    return pose_pt(f, ELBOW[side])


def wrist_px(f, side):
    """Use YOLO wrist for stable arm drift; fall back to MediaPipe when pose is missing."""
    pose = pose_pt(f, WRIST[side])
    if pose is not None:
        return pose
    h = get_hand(f, side)
    return h["wrist"] if h is not None else pose_pt(f, WRIST[side])


class ActionContext:
    """一次会话的上下文和最近可靠的前臂几何量。"""

    def __init__(self, name, side, forearm_cm, target_reps):
        self.name = name
        self.affected = side                      # 'left' / 'right'
        self.healthy = "left" if side == "right" else "right"
        self.forearm_cm = float(forearm_cm)
        self.target_reps = int(target_reps)
        self.forearm_px = None
        self.elbow_point = None
        self.wrist_point = None
        self.elbow_stale = False

    def update_forearm(self, pose):
        """用患侧肘腕更新前臂长度；肘短暂丢失时保留最近可靠点。"""
        pe = pose_pt({"pose": pose}, ELBOW[self.affected])
        pw = pose_pt({"pose": pose}, WRIST[self.affected])
        if pw is not None:
            self.wrist_point = pw if self.wrist_point is None else 0.8 * self.wrist_point + 0.2 * pw
        if pe is None:
            self.elbow_stale = self.elbow_point is not None
            pe = self.elbow_point
        else:
            self.elbow_point = pe if self.elbow_point is None else 0.8 * self.elbow_point + 0.2 * pe
            self.elbow_stale = False
        if pe is None or pw is None:
            return
        d = dist(pe, pw)
        if d < 10:
            return
        self.forearm_px = d if self.forearm_px is None else 0.8 * self.forearm_px + 0.2 * d

    def to_cm(self, px):
        if not self.forearm_px:
            return None
        return px / self.forearm_px * self.forearm_cm


def hand_is_open(hand):
    feature = hand_features(hand["pts"])
    angles = feature["angles"]
    strict = fingers_open(angles, {f: {"mcp": FIST_TEMPLATE["open_angle"], "pip": FIST_TEMPLATE["open_angle"]} for f in FOUR})
    return strict or (feature["curl"] >= FIST_TEMPLATE["open_angle"] and feature["spread"] >= 0.35)


def hand_is_closed(hand):
    feature = hand_features(hand["pts"])
    angles = feature["angles"]
    strict = fingers_closed(angles, {f: {"mcp": FIST_TEMPLATE["closed_angle"], "pip": FIST_TEMPLATE["closed_angle"]} for f in FOUR})
    return strict or (feature["curl"] <= FIST_TEMPLATE["closed_angle"] and feature["tip_ratio"] <= 1.25)


def frame_size(f):
    size = f.get("frame_size", (1920, 1080))
    return float(size[0]), float(size[1])


def tracking_guard(action, hands):
    states = [hand_state(h) for h in hands if h is not None]
    if "lost" in states:
        action.tracking_state = "手部丢失"
        return False
    action.tracking_state = "遮挡桥接" if "bridged" in states else "正常"
    return True


def wrist_for(ctx, f, side):
    return ctx.wrist_point if side == ctx.affected and ctx.wrist_point is not None else wrist_px(f, side)


def hand_event_ok(duration, limits, drift=0.0, drift_limit=1.0):
    return limits[0] <= duration <= limits[1] and drift <= drift_limit


class Action:
    key = ""
    name = ""
    desc = ""
    has_mode = False

    def __init__(self, ctx):
        self.ctx = ctx
        self.reset()

    def reset(self):
        self.reps = []
        self.count = 0
        self.passed = 0
        self.hint = ""
        self.phase = "idle"
        self.live = ""
        self.tracking_state = "正常"

    def update(self, f):
        raise NotImplementedError

    def status(self):
        return {
            "name": self.name,
            "hint": self.hint,
            "live": self.live,
            "phase": self.phase,
            "tracking": self.tracking_state,
            "count": self.count,
            "passed": self.passed,
            "target": self.ctx.target_reps,
            "guidance_rois": self.template.get("guidance_rois", {}) if hasattr(self, "template") else {},
            "active_guidance": getattr(self, "active_guidance", ""),
            "position_hint": getattr(self, "position_hint", ""),
            "baseline_status": getattr(self, "baseline_status", ""),
        }

    def summary(self):
        durs = [r["duration"] for r in self.reps]
        return {
            "name": self.name,
            "target": self.ctx.target_reps,
            "count": self.count,
            "passed": self.passed,
            "rate": (self.passed / self.count * 100) if self.count else 0.0,
            "total": sum(durs),
            "avg": (sum(durs) / len(durs)) if durs else 0.0,
            "review": sum(1 for r in self.reps if not r.get("ok")),
            "failures": [r for r in self.reps if not r.get("ok")],
            "reps": self.reps,
        }

    def _record(self, rep):
        self.reps.append(rep)
        self.count += 1
        if rep["ok"]:
            self.passed += 1
