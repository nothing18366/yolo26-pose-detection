"""关键点几何与手指关节角计算。

角度约定：三点余弦定理，180° = 完全伸直，0° = 完全折叠。
坐标一律为像素 (x, y)，所有函数都是纯函数，方便单独测试。
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


def finger_axis(pts):
    """四指 MCP->TIP 的平均方向（单位向量），作为手指长轴。"""
    pts = np.asarray(pts, dtype=float)
    acc = np.zeros(2)
    for f in FOUR:
        mcp, _pip, _dip, tip = FINGERS[f]
        acc += unit(pts[tip] - pts[mcp])
    return unit(acc) if np.linalg.norm(acc) > 1e-6 else np.array([0.0, 1.0])


def forearm_axis(elbow, wrist):
    """前臂轴单位向量（手腕 -> 肘部，即摩擦动作的正方向）与前臂像素长度。"""
    v = np.asarray(elbow, dtype=float) - np.asarray(wrist, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n if n > 1e-6 else np.zeros(2)), n
