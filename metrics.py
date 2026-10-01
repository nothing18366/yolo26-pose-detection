"""关键点几何与手指关节角计算。

角度约定：三点余弦定理，180° = 完全伸直，0° = 完全折叠。
图像坐标使用像素 (x, y)，单指形状特征优先使用 MediaPipe 三维手部坐标。
"""

import math

import numpy as np

# COCO-17 关键点索引
ELBOW = {"left": 7, "right": 8}
WRIST = {"left": 9, "right": 10}

# MediaPipe 手部 21 点：拇指 1-4，食指 5-8，中指 9-12，无名指 13-16，小指 17-20
HAND_WRIST = 0
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FOUR = tuple(FINGERS)  # 四指（不含拇指），阈值判定只看这四指
FINGERTIPS = (8, 12, 16, 20)
JOINTS = ("mcp", "pip")


def local_frame(elbow, wrist):
    """Return wrist->elbow axis and its perpendicular for viewpoint-normalized metrics."""
    axis, length = forearm_axis(elbow, wrist)
    perp = np.array([-axis[1], axis[0]])
    return axis, perp, length


def project_local(point, origin, axis, perp):
    """Project an image point to (along-axis, lateral) coordinates."""
    d = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    return float(np.dot(d, axis)), float(np.dot(d, perp))


class ExpSmoother:
    """Small EMA used to prevent one noisy landmark from changing a phase."""
    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.value = None

    def update(self, value):
        value = np.asarray(value, dtype=float)
        self.value = value if self.value is None else self.alpha * value + (1 - self.alpha) * self.value
        return self.value.copy()


def angle_at(a, b, c):
    """顶点 b 处的夹角，返回 0~180 度。"""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def line_angle(v1, v2):
    """两条直线方向的夹角（0~90 度），忽略正负方向。

    注意：向量夹角要把顶点放在原点上，即 angle_at(v1, 原点, v2)。
    """
    a = angle_at(v1, (0.0, 0.0), v2)
    return min(a, 180.0 - a)


def signed_angle(v1, v2):
    """Signed 2-D angle from v1 to v2 in degrees (-180, 180)."""
    a, b = unit(v1), unit(v2)
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(float(a[0] * b[1] - a[1] * b[0]), float(np.dot(a, b))))


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)


def finger_angles(pts):
    """pts: 21x2 像素坐标 -> {手指: {"mcp": 度, "pip": 度}}"""
    pts = np.asarray(pts, dtype=float)
    wrist = pts[HAND_WRIST]
    out = {}
    for name, (mcp, pip, dip, _tip) in FINGERS.items():
        out[name] = {
            "mcp": angle_at(wrist, pts[mcp], pts[pip]),
            "pip": angle_at(pts[mcp], pts[pip], pts[dip]),
        }
    return out


def curl_metric(angles):
    """四指 mcp/pip 的平均角，作为一个整体"手指伸展程度"标量（180=全伸直）。"""
    vals = [angles[f][j] for f in FOUR for j in JOINTS]
    return float(np.mean(vals)) if vals else 0.0


def fingers_open(angles, thresholds):
    """thresholds: {手指: {"mcp": 下限, "pip": 下限}}，四指全部达标才算张开。"""
    return all(angles[f][j] >= thresholds[f][j] for f in FOUR for j in JOINTS)


def fingers_closed(angles, thresholds):
    """thresholds: {手指: {"mcp": 上限, "pip": 上限}}，四指全部达标才算握紧。"""
    return all(angles[f][j] <= thresholds[f][j] for f in FOUR for j in JOINTS)


def palm_center(pts):
    """掌心近似：四个 MCP 的平均点。"""
    pts = np.asarray(pts, dtype=float)
    return np.mean([pts[FINGERS[f][0]] for f in FOUR], axis=0)


def fingertips(pts):
    pts = np.asarray(pts, dtype=float)
    return pts[list(FINGERTIPS)]


def hand_features(pts):
    """Stable scalar features for open/closed hand classification."""
    pts = np.asarray(pts, dtype=float)
    angles = finger_angles(pts)
    palm = palm_center(pts)
    length = hand_length(pts)
    tip_ratio = float(np.mean([dist(pts[i], palm) for i in FINGERTIPS]) / max(length, 1e-6))
    spread = float(np.mean([dist(pts[i], pts[j]) for i, j in ((8, 12), (12, 16), (16, 20))]) / max(length, 1e-6))
    return {"angles": angles, "curl": curl_metric(angles), "tip_ratio": tip_ratio, "spread": spread}


def finger_extension_features(pts):
    """Normalized thumb/index/four-finger distances for small hand motions."""
    pts = np.asarray(pts, dtype=float)
    palm = palm_center(pts)
    scale = max(hand_length(pts), 1e-6)
    return {
        "thumb": float(np.linalg.norm(pts[4] - pts[5]) / scale),
        "index": float(np.linalg.norm(pts[8] - pts[9]) / scale),
        "middle": float(np.linalg.norm(pts[12] - pts[9]) / scale),
        "ring": float(np.linalg.norm(pts[16] - pts[13]) / scale),
        "pinky": float(np.linalg.norm(pts[20] - pts[17]) / scale),
        "thumb_palm": float(np.linalg.norm(pts[4] - palm) / scale),
        "index_palm": float(np.linalg.norm(pts[8] - palm) / scale),
    }


def isolated_finger_features(hand):
    """Hand-local shape ratios; use the model's 3D hand coordinates when available."""
    source = hand.get("world")
    pts = np.asarray(source if source is not None else hand["pts"], dtype=float)
    length = np.linalg.norm(pts[9] - pts[0])
    width = np.linalg.norm(pts[5] - pts[17])
    if length < 1e-6 or width < 1e-6 or not np.isfinite(pts).all():
        return None
    if source is None and length < 0.5 * width:
        return None
    across = (pts[5] - pts[17]) / width
    reach = {name: dist(pts[tip], pts[0]) / max(dist(pts[mcp], pts[0]), 1e-6)
             for name, (mcp, _, _, tip) in FINGERS.items()}
    return {
        "thumb": float(np.dot(pts[4] - pts[2], across) / length),
        "thumb_gap": dist(pts[4], pts[5]) / length,
        "index": reach["index"],
        "index_angle": angle_at(pts[5], pts[6], pts[7]),
        "other_reach": float(np.median([reach[name] for name in ("middle", "ring", "pinky")])),
        "source": "world" if source is not None else "image",
    }


def stretch_features(affected, healthy, pose=None):
    """Contact and motion proxies in hand/body coordinates, without a table ROI."""
    pts = np.asarray(affected["pts"], dtype=float)
    targets = np.asarray(healthy["pts"], dtype=float)
    length = hand_length(pts)
    if pose is not None:
        side = affected.get("side")
        if side in ELBOW:
            e, w = ELBOW[side], WRIST[side]
            if pose[e, 2] >= .4 and pose[w, 2] >= .4:
                length = max(length, .35 * dist(pose[e, :2], pose[w, :2]))
    if length < 5 or not np.isfinite(pts).all():
        return None
    vertical = np.array([0.0, 1.0])
    if pose is not None and all(pose[i, 2] >= .5 for i in (5, 6, 11, 12)):
        vertical = unit((pose[11, :2] + pose[12, :2]) - (pose[5, :2] + pose[6, :2]))
    directions = [unit(pts[tip] - pts[mcp]) for mcp, _, _, tip in FINGERS.values()]
    direction = float(np.median([np.dot(v, vertical) for v in directions]))
    assisting_direction = (float(np.median([np.dot(unit(targets[tip] - targets[mcp]), vertical)
                                            for mcp, _, _, tip in FINGERS.values()]))
                            if len(targets) == 21 else None)
    gaps = np.array([np.min(np.linalg.norm(targets - tip, axis=1)) for tip in fingertips(pts)]) / length
    motion = {"curl": curl_metric(finger_angles(pts))}
    if affected.get("world") is not None:
        world = np.asarray(affected["world"], dtype=float)
        forward = world[9] - world[0]
        motion["pitch"] = float(np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2]))))
    if pose is not None and affected.get("side") in WRIST:
        side = affected["side"]
        wrist_i, other_i = WRIST[side], WRIST["right" if side == "left" else "left"]
        elbow_i = ELBOW[side]
        if all(pose[i, 2] >= .5 for i in (wrist_i, other_i, elbow_i)):
            forearm = dist(pose[elbow_i, :2], pose[wrist_i, :2])
            if (forearm >= 20 and dist(pts[0], pose[wrist_i, :2]) <= length and
                    dist(targets[0], pose[other_i, :2]) <= length):
                motion["wrist_gap"] = 100.0 * dist(pose[wrist_i, :2], pose[other_i, :2]) / forearm
    return {"direction": direction, "assisting_direction": assisting_direction,
            "gap": float(np.median(gaps)), "length": length,
            "motion": motion}


def roi_contains(point, roi, width, height, tolerance=0.0):
    """Check a normalized [x0, y0, x1, y1] ROI against pixel coordinates."""
    x0, y0, x1, y1 = roi
    px, py = np.asarray(point, dtype=float)
    tx, ty = tolerance * width, tolerance * height
    return x0 * width - tx <= px <= x1 * width + tx and y0 * height - ty <= py <= y1 * height + ty


def gesture_signature(pts, angles=None):
    """Binary finger-open signature used by the video-defined gesture sequence."""
    angles = angles or finger_angles(pts)
    return tuple(int(angles[name]["pip"] >= 145.0 and angles[name]["mcp"] >= 145.0)
                 for name in ("index", "middle", "ring", "pinky")) + (
        int(np.linalg.norm(np.asarray(pts[4]) - np.asarray(pts[0])) > hand_length(pts) * 0.55),
    )


def hand_length(pts):
    pts = np.asarray(pts, dtype=float)
    wrist = pts[HAND_WRIST]
    mcp = np.mean([pts[FINGERS[f][0]] for f in FOUR], axis=0)
    return float(np.linalg.norm(mcp - wrist))


def palm_normal_z(xyz):
    """Signed 3D palm normal; used only to distinguish palm-up/down in action 3."""
    if xyz is None or len(xyz) < 18:
        return None
    xyz = np.asarray(xyz, dtype=float)
    a = xyz[5] - xyz[0]
    b = xyz[17] - xyz[0]
    return float(np.cross(a, b)[2])


def forearm_axis(elbow, wrist):
    """前臂轴单位向量（手腕 -> 肘部，即摩擦动作的正方向）与前臂像素长度。"""
    v = np.asarray(elbow, dtype=float) - np.asarray(wrist, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n if n > 1e-6 else np.zeros(2)), n
