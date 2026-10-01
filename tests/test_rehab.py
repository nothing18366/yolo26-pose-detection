"""可运行自检：几何/角度 + 三个动作状态机 + Qt 界面联调（合成关键点，不需要摄像头）。

运行：python tests/test_rehab.py
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无摄像头/无显示器也能跑界面联调
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from metrics import (
    ELBOW,
    FINGERS,
    FOUR,
    WRIST,
    angle_at,
    curl_metric,
    finger_angles,
    fingers_closed,
    fingers_open,
    fingertips,
    line_angle,
    palm_center,
)
from actions import ELBOW as A_ELBOW, WRIST as A_WRIST  # noqa: F401  (保持与实现同一份索引)
from actions import (ActionContext, CupTransfer, FingerAbduction,
                     FingerPress, FingerStretch, FistOpenClose, RubBackOfHand)
from video_templates import load as load_template
from tracking import HandTracker


def rotate(v, rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def synth_hand(mcp_deg=170.0, pip_deg=170.0, origin=(0.0, 0.0), seg=(70.0, 50.0, 30.0, 25.0)):
    """构造 21 点手：四指按给定 MCP/PIP 角弯曲；拇指只占位（不参与阈值判定）。"""
    pts = np.zeros((21, 2))
    wrist = np.array(origin, dtype=float)
    pts[0] = wrist
    for i in range(1, 5):
        pts[i] = wrist + np.array([-25.0 * i, 20.0 * i])
    for k, f in enumerate(FOUR):
        mcp_i, pip_i, dip_i, tip_i = FINGERS[f]
        ang = math.radians((k - 1.5) * 8.0)
        u = np.array([math.sin(ang), -math.cos(ang)])
        mcp = wrist + u * seg[0]
        v1 = rotate(u, math.radians(180.0 - mcp_deg))
        pip = mcp + v1 * seg[1]
        v2 = rotate(v1, math.radians(180.0 - pip_deg))
        dip = pip + v2 * seg[2]
        tip = dip + v2 * seg[3]
        pts[mcp_i], pts[pip_i], pts[dip_i], pts[tip_i] = mcp, pip, dip, tip
    return pts


def hand_at(side, palm_pos, mcp_deg=170.0, pip_deg=170.0):
    pts = synth_hand(mcp_deg, pip_deg)
    pts = pts + (np.asarray(palm_pos, float) - palm_center(pts))
    return {
        "side": side,
        "pts": pts,
        "angles": finger_angles(pts),
        "palm": palm_center(pts),
        "wrist": pts[0].copy(),
    }


def frame(t, pose=None, hands=()):
    return {"t": t, "pose": pose, "hands": list(hands)}


def arm_pose(side, wrist, elbow):
    pose = np.zeros((17, 3))
    pose[ELBOW[side]] = [elbow[0], elbow[1], 1.0]
    pose[WRIST[side]] = [wrist[0], wrist[1], 1.0]
    return pose


# ---------------- 几何 / 角度 ----------------
def test_geometry():
    assert abs(angle_at((0, 0), (1, 0), (2, 0)) - 180.0) < 1e-6
    assert abs(angle_at((0, 0), (1, 0), (1, 1)) - 90.0) < 1e-6
    assert abs(angle_at((1, 0), (0, 0), (1, 1)) - 45.0) < 1e-6
    assert abs(line_angle((1, 0), (-1, 0)) - 0.0) < 1e-6
    assert abs(line_angle((1, 0), (0, 1)) - 90.0) < 1e-6

    a = finger_angles(synth_hand(175.0, 175.0))
    for f in FOUR:
        assert abs(a[f]["mcp"] - 175.0) < 1e-6, (f, a[f])
        assert abs(a[f]["pip"] - 175.0) < 1e-6, (f, a[f])
    assert fingers_open(a, {f: {"mcp": 150.0, "pip": 150.0} for f in FOUR})
    assert not fingers_closed(a, {f: {"mcp": 100.0, "pip": 100.0} for f in FOUR})

    b = finger_angles(synth_hand(80.0, 60.0))
    assert fingers_closed(b, {f: {"mcp": 100.0, "pip": 100.0} for f in FOUR})
    assert not fingers_open(b, {f: {"mcp": 150.0, "pip": 150.0} for f in FOUR})


# ---------------- 动作2 握拳伸展 ----------------
def fist_frames(seq):
    """seq: [(t, 患侧开?, 健侧开?)]"""
    out = []
    for t, a_open, h_open in seq:
        out.append(
            frame(
                t,
                None,
                [
                    hand_at("right", (600, 600), *( (175.0, 175.0) if a_open else (80.0, 60.0) )),
                    hand_at("left", (400, 600), *( (175.0, 175.0) if h_open else (80.0, 60.0) )),
                ],
            )
        )
    return out


def test_fist_counts_one_cycle():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FistOpenClose(ctx)
    for f in fist_frames([(0.0, 1, 1), (0.5, 1, 1), (0.9, 0, 0), (1.4, 0, 0), (2.0, 1, 1)]):
        act.update(f)
    assert act.count == 1, act.count
    assert act.passed == 1, act.reps
    assert 1.0 <= act.reps[0]["duration"] <= 3.0


def test_fist_rejects_slow_cycle():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FistOpenClose(ctx)
    for f in fist_frames([(0.0, 1, 1), (0.5, 0, 0), (2.0, 0, 0), (4.0, 1, 1)]):
        act.update(f)
    assert act.count == 1, act.count
    assert act.passed == 0, act.reps
    assert act.reps[0]["duration"] > 3.0


def test_fist_uses_fixed_template_threshold():
    """动作2不读取患者基线，患侧未达到模板张开阈值时不启动周期。"""
    seq = [(0.0, 115.0, 1), (0.4, 115.0, 1), (0.9, 95.0, 0), (1.4, 95.0, 0), (1.9, 115.0, 1)]
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FistOpenClose(ctx)
    for t, a_deg, healthy_open in seq:
        act.update(frame(t, None, [
            hand_at("right", (600, 600), a_deg, a_deg),
            hand_at("left", (400, 600), *((175.0, 175.0) if healthy_open else (80.0, 60.0))),
        ]))
    assert act.count == 0, act.reps


# ---------------- 动作1 摩擦手背 ----------------
def rub_pose():
    return arm_pose("right", (500.0, 700.0), (500.0, 400.0))  # 前臂长 300px，轴朝上


def rub_frames(travel):
    out, t = [], 0.0
    for i in range(11):
        out.append(frame(t, rub_pose(), [hand_at("left", (500.0, 700.0 - travel * i / 10))]))
        t += 0.1
    for i in range(1, 11):
        out.append(frame(t, rub_pose(), [hand_at("left", (500.0, 700.0 - travel + travel * i / 10))]))
        t += 0.1
    return out


def test_rub_counts_qualified_round_trip():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    assert abs(ctx.forearm_px - 300.0) < 1e-6
    act = RubBackOfHand(ctx)
    for f in rub_frames(180.0):  # 0.6 前臂长
        act.update(f)
    assert act.count == 1, (act.count, act.reps)
    assert act.passed == 1, act.reps


def test_rub_short_trip_is_not_counted():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    act = RubBackOfHand(ctx)
    for f in rub_frames(90.0):  # 0.3 前臂长，不够 0.6
        act.update(f)
    assert act.count == 0, act.reps


def test_rub_bridged_partial_cycle_not_confirmed():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    act = RubBackOfHand(ctx)
    for i, p in enumerate([(500.0, 700.0), (500.0, 600.0), (500.0, 520.0)]):
        act.update(frame(i * 0.1, rub_pose(), [{**hand_at("left", p), "state": "fresh"}]))
    bridged = {**hand_at("left", (500.0, 450.0)), "state": "bridged", "predicted": True}
    act.update(frame(0.4, rub_pose(), [bridged]))
    assert act.count == 0 and act.tracking_state == "遮挡桥接", act.reps


def test_rub_bridged_hand_can_finish_round_trip():
    """桥接帧继续推进去程/回程，完整周期仍可确认。"""
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    act = RubBackOfHand(ctx)
    seq = [
        (0.0, (500.0, 700.0), "fresh"),
        (0.1, (500.0, 540.0), "fresh"),
        (0.2, (500.0, 480.0), "fresh"),
        (0.3, (500.0, 470.0), "bridged"),
        (0.4, (500.0, 700.0), "bridged"),
        (0.45, (500.0, 700.0), "bridged"),
        (0.5, (500.0, 700.0), "bridged"),
    ]
    for t, p, state in seq:
        hand = {**hand_at("left", p), "state": state, "predicted": state == "bridged"}
        act.update(frame(t, rub_pose(), [hand]))
    assert act.count == 1 and act.passed == 1, act.reps


def test_rub_static_hand_not_counted():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    act = RubBackOfHand(ctx)
    for i in range(20):
        act.update(frame(i * 0.1, rub_pose(), [hand_at("left", (500.0, 700.0))]))
    assert act.count == 0, act.reps


def test_rub_sideways_not_rubbing():
    """停在手背上没动，抬起来不应计数。"""
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.update_forearm(rub_pose())
    act = RubBackOfHand(ctx)
    for i in range(5):
        act.update(frame(i * 0.1, rub_pose(), [hand_at("left", (500.0, 700.0))]))
    act.update(frame(0.6, rub_pose(), [hand_at("left", (900.0, 700.0))]))  # 移开
    assert act.count == 0, act.reps


def test_rub_elbow_short_loss_uses_cached_axis():
    ctx = ActionContext("测试", "right", 25.0, 10)
    pose = rub_pose()
    ctx.update_forearm(pose)
    act = RubBackOfHand(ctx)
    for i, p in enumerate([(500.0, 700.0), (500.0, 620.0), (500.0, 520.0)]):
        act.update(frame(i * 0.1, pose, [hand_at("left", p)]))
    missing = pose.copy()
    missing[8, 2] = 0.0
    ctx.update_forearm(missing)
    act.update(frame(0.35, missing, [hand_at("left", (500.0, 500.0))]))
    assert ctx.elbow_stale and act.cur is not None, (ctx.elbow_stale, act.phase)


def test_rub_counts_with_moving_free_arm():
    ctx = ActionContext("测试", "right", 25.0, 10)
    act = RubBackOfHand(ctx)
    for i in range(21):
        theta = math.radians(45.0 * i / 20)
        wrist = np.array([500.0 + 100.0 * i / 20, 700.0 + 30.0 * i / 20])
        axis = rotate((0.0, -1.0), theta)
        elbow = wrist + 300.0 * axis
        travel = 0.75 * (i / 10 if i <= 10 else (20 - i) / 10)
        palm = wrist + 300.0 * (travel * axis + 0.2 * rotate(axis, math.pi / 2))
        pose = arm_pose("right", wrist, elbow)
        pose[5] = [300.0, 300.0, 1.0]
        pose[6] = [500.0 + 40.0 * i / 20, 250.0, 1.0]
        ctx.update_forearm(pose)
        act.update(frame(i * 0.1, pose, [hand_at("left", palm)]))
    assert act.count == 1 and act.passed == 1, act.reps


def test_rub_rejects_free_arm_motion_without_relative_stroke():
    ctx = ActionContext("测试", "right", 25.0, 10)
    act = RubBackOfHand(ctx)
    for i in range(21):
        theta = math.radians(45.0 * i / 20)
        wrist = np.array([500.0 + 100.0 * i / 20, 700.0 + 30.0 * i / 20])
        axis = rotate((0.0, -1.0), theta)
        pose = arm_pose("right", wrist, wrist + 300.0 * axis)
        ctx.update_forearm(pose)
        act.update(frame(i * 0.1, pose, [hand_at("left", wrist + 60.0 * axis)]))
    assert act.count == 0, act.reps


def test_rub_wrong_affected_side_does_not_count_stationary_hand():
    ctx = ActionContext("测试", "left", 25.0, 10)
    act = RubBackOfHand(ctx)
    for i in range(21):
        shift = 180.0 * (i / 10 if i <= 10 else (20 - i) / 10)
        pose = arm_pose("left", (500.0, 700.0 + shift), (500.0, 400.0 + shift))
        pose[5] = [500.0, 200.0, 1.0]
        pose[6] = [800.0, 200.0, 1.0]
        ctx.update_forearm(pose)
        act.update(frame(i * 0.1, pose, [hand_at("right", (500.0, 700.0))]))
    assert act.count == 0, act.reps


# ---------------- 动作3 牵伸手指 ----------------
def stretch_pair(mode, degree, side="right", contact=True):
    affected = hand_at(side, (600.0, 620.0), degree, degree)
    if mode == "flex":
        affected["pts"] = 2 * affected["palm"] - affected["pts"]
        affected["palm"] = palm_center(affected["pts"])
        affected["wrist"] = affected["pts"][0].copy()
    healthy_side = "left" if side == "right" else "right"
    healthy = hand_at(healthy_side, (0.0, 0.0))
    pts = healthy["pts"]
    pts = (pts - pts[0]) @ np.array([[0.0, -1.0], [1.0, 0.0]])
    target = fingertips(affected["pts"]).mean(axis=0) if contact else np.array([1200.0, 300.0])
    pts += target - fingertips(pts).mean(axis=0)
    healthy["pts"], healthy["palm"], healthy["wrist"] = pts, palm_center(pts), pts[0].copy()
    return [affected, healthy]


def stretch_frames(mode, deg_from, deg_to, hold=1.0, side="right"):
    degrees = [deg_from] * 8 + [deg_to] * max(1, int(hold / .1)) + [deg_from] * 8
    return [frame(i * .1, None, stretch_pair(mode, deg, side)) for i, deg in enumerate(degrees)]


def test_stretch_flex_counts_press():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 10))
    for f in stretch_frames("flex", 175.0, 120.0):
        act.update(f)
    assert act.count == 1 and act.passed == 1, act.reps
    assert len(act.modes["flex"]) == 1 and not act.modes["ext"]


def test_stretch_ext_needs_only_small_change():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 10))
    for f in stretch_frames("ext", 170.0, 178.0):
        act.update(f)
    assert act.count == 1 and act.passed == 1, act.reps
    assert len(act.modes["ext"]) == 1 and not act.modes["flex"]


def test_stretch_too_short_not_counted():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 10))
    for f in stretch_frames("flex", 175.0, 120.0, hold=.1):
        act.update(f)
    assert act.count == 0, act.reps


def test_stretch_unlimited_unequal_counts_and_auto_switch():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 5))
    t = 0.0
    for mode in ["ext"] * 20 + ["flex"] * 3:
        for f in stretch_frames(mode, 175.0, 140.0):
            f["t"] += t
            act.update(f)
        t = f["t"] + .1
        for _ in range(5):
            act.update(frame(t, None, stretch_pair(mode, 175.0, contact=False)))
            t += .1
    assert act.count == 23, act.reps
    assert len(act.modes["ext"]) == 20 and len(act.modes["flex"]) == 3


def test_stretch_static_contact_and_long_hold_do_not_repeat():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 20))
    for i in range(80):
        act.update(frame(i * .1, None, stretch_pair("ext", 175.0)))
    assert act.count == 0
    for i in range(80):
        act.update(frame(8 + i * .1, None, stretch_pair("ext", 140.0)))
    assert act.count == 0


def test_stretch_prediction_does_not_confirm_return():
    act = FingerStretch(ActionContext("测试", "right", 25.0, 20))
    for f in stretch_frames("ext", 175.0, 140.0)[:18]:
        act.update(f)
    for i in range(5):
        hands = [{**h, "state": "bridged"} for h in stretch_pair("ext", 175.0)]
        act.update(frame(1.8 + i * .1, None, hands))
    assert act.count == 0
    for i in range(4):
        act.update(frame(2.3 + i * .1, None, stretch_pair("ext", 175.0)))
    assert act.count == 1, act.reps


def test_stretch_left_side_matches_right_and_ignores_camera_translation():
    for side in ("left", "right"):
        act = FingerStretch(ActionContext("测试", side, 25, 10))
        for i, f in enumerate(stretch_frames("flex", 175.0, 140.0, side=side)):
            for h in f["hands"]:
                offset = np.array([i * 4.0, i * 2.0])
                h["pts"] += offset
                h["palm"] += offset
                h["wrist"] += offset
            act.update(f)
        assert len(act.modes["flex"]) == 1 and act.passed == 1, act.reps


def test_stretch_flex_uses_consistent_pose_wrists_when_finger_shape_is_constant():
    act = FingerStretch(ActionContext("测试", "right", 25, 10))
    for i, shift in enumerate([0.0] * 8 + [60.0] * 10 + [0.0] * 8):
        hands = stretch_pair("flex", 175.0)
        hands[1]["pts"] += np.array([shift, 0.0])
        hands[1]["wrist"] = hands[1]["pts"][0].copy()
        pose = arm_pose("right", hands[0]["wrist"], hands[0]["wrist"] - np.array([0.0, 300.0]))
        pose[9] = [*hands[1]["wrist"], 1.0]
        act.update(frame(i * .1, pose, hands))
    assert act.count == 1 and act.reps[0]["feature"] == "wrist_gap", act.reps


def test_stretch_rejects_disagreement_between_pose_and_hand_wrists():
    act = FingerStretch(ActionContext("测试", "right", 25, 10))
    for i, shift in enumerate([0.0] * 8 + [60.0] * 10 + [0.0] * 8):
        hands = stretch_pair("flex", 175.0)
        pose = arm_pose("right", hands[0]["wrist"], hands[0]["wrist"] - np.array([0.0, 300.0]))
        pose[9] = [*(hands[1]["wrist"] + np.array([300.0 + shift, 0.0])), 1.0]
        act.update(frame(i * .1, pose, hands))
    assert act.count == 0, act.reps


def test_stretch_real_assisting_wrist_keeps_cycle_but_requires_review():
    act = FingerStretch(ActionContext("测试", "right", 25, 10))
    degrees = [175.0] * 8 + [140.0] * 10 + [175.0] * 8
    for i, degree in enumerate(degrees):
        hands = stretch_pair("flex", degree)
        target = fingertips(hands[0]["pts"]).mean(axis=0)
        hands[1]["pts"] += target - hands[1]["pts"][0]
        hands[1]["wrist"] = target.copy()
        pose = arm_pose("right", hands[0]["wrist"], hands[0]["wrist"] - np.array([0.0, 300.0]))
        pose[9] = [*target, 1.0]
        observed = hands if i < 8 else [hands[0]]
        act.update(frame(i * .1, pose, observed))
    assert act.count == 1 and act.passed == 0, act.reps
    assert act.reps[0]["wrist_assistance_ratio"] > 0.4


def test_hand_tracker_keeps_side_during_occlusion():
    tracker = HandTracker(max_missing_s=0.5)
    left = hand_at("left", (400, 500))
    right = hand_at("right", (700, 500))
    raw = [{"cat": "Left", "pts": left["pts"]}, {"cat": "Right", "pts": right["pts"]}]
    first = tracker.update(raw, 0.0)
    assert {h["side"] for h in first} == {"left", "right"}
    missing = tracker.update([{"cat": "Right", "pts": right["pts"]}], 0.2)
    assert {h["side"] for h in missing} == {"left", "right"}
    assert any(h.get("stale") for h in missing)
    bridged = next(h for h in missing if h["state"] != "fresh")
    assert bridged["state"] == "bridged" and bridged["predicted"]
    lost = tracker.update([{"cat": "Right", "pts": right["pts"]}], 0.6)
    lost_hand = next(h for h in lost if h["state"] == "lost")
    assert lost_hand["state"] == "lost"


def test_hand_tracker_reacquires_sides_from_pose_wrists():
    tracker = HandTracker()
    pose = np.zeros((17, 3))
    pose[9] = [550, 700, 0.99]
    pose[10] = [250, 700, 0.99]
    left = hand_at("left", (550, 700))
    right = hand_at("right", (250, 700))
    raw = [{"pts": left["pts"]}, {"pts": right["pts"]}]
    tracker.update(raw, 0.0, pose)
    tracker.last["left"], tracker.last["right"] = tracker.last["right"], tracker.last["left"]
    hands = tracker.update(raw, 0.1, pose)
    actual = {h["side"]: h["pts"][0] for h in hands if h["state"] == "fresh"}
    assert np.linalg.norm(actual["left"] - left["pts"][0]) < 1
    assert np.linalg.norm(actual["right"] - right["pts"][0]) < 1


def test_cup_counts_lift_and_place_independently_past_eight():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = CupTransfer(ctx)
    t = 0.0
    for y in [700.0] * 6:
        pose = arm_pose("right", (500, y), (500, 400))
        act.update(frame(t, pose, [hand_at("right", (500, y))]))
        t += 0.2
    for _ in range(10):
        for y in [600.0] * 3 + [730.0] * 3:
            pose = arm_pose("right", (500, y), (500, 400))
            act.update(frame(t, pose, [hand_at("right", (500, y))]))
            t += 0.2
        for _ in range(6):
            pose = arm_pose("right", (500, 730), (500, 400))
            act.update(frame(t, pose, [hand_at("right", (500, 730))]))
            t += 0.2
    assert act.lift_count == 10 and act.place_count == 10, (act.lift_count, act.place_count)


def cup_pose(side, y, shift=(0, 0), scale=1.0, angle=0.0):
    pose = arm_pose(side, (500, y), (300, 650))
    pose[5], pose[6] = [300, 300, 1], [500, 300, 1]
    pose[11], pose[12] = [300, 900, 1], [500, 900, 1]
    for point in pose:
        if point[2]:
            point[:2] = rotate(point[:2], angle) * scale + shift
    return pose


def test_cup_pose_only_mirror_roll_translation_and_zoom():
    for side in ("left", "right"):
        act = CupTransfer(ActionContext("测试", side, 25, 10))
        act.update(frame(-.1, None, []))
        ys = [700] * 10 + [590] * 10 + [700] * 10
        for i, y in enumerate(ys):
            pose = cup_pose(side, y, (i * 7, -i * 5), 1 + i * .02, i * .02)
            # Healthy hand points deliberately jump/disappear; no affected hand is detected.
            hands = [hand_at(act.ctx.healthy, (i * 200, i * 150))] if i % 2 else []
            act.update(frame(i * .1, pose, hands))
        assert (act.lift_count, act.place_count) == (1, 1), act.reps


def test_cup_body_motion_and_wrist_jitter_do_not_count():
    act = CupTransfer(ActionContext("测试", "right", 25, 10))
    for i in range(50):
        pose = cup_pose("right", 700 + (i % 3 - 1) * 3,
                        (i * 7, -i * 5), 1 + i * .02, i * .02)
        act.update(frame(i * .1, pose, []))
    assert (act.lift_count, act.place_count) == (0, 0), act.reps


def test_cup_holding_and_point_loss_never_invent_a_place():
    for missing_duration in (.2, .8):
        act = CupTransfer(ActionContext("测试", "right", 25, 10))
        for i, y in enumerate([700] * 10 + [590] * 20):
            act.update(frame(i * .1, cup_pose("right", y), []))
        assert (act.lift_count, act.place_count) == (1, 0)
        act.update(frame(3.0, None, []))
        for i in range(10):
            act.update(frame(3.0 + missing_duration + i * .1, cup_pose("right", 700), []))
        assert act.lift_count == 1
        assert act.place_count == (1 if missing_duration < .35 else 0), act.reps


def test_cup_horizontal_fetch_does_not_count():
    act = CupTransfer(ActionContext("测试", "right", 25, 10))
    for i in range(10):
        act.update(frame(i * .1, cup_pose("right", 700), []))
    for i in range(10):
        pose = cup_pose("right", 640)
        pose[10, 0] += 200
        act.update(frame(1.0 + i * .1, pose, []))
    assert (act.lift_count, act.place_count) == (0, 0), act.reps


def test_cup_small_drop_while_raised_does_not_split_cycle():
    act = CupTransfer(ActionContext("测试", "right", 25, 10))
    ys = [700] * 10 + [500] * 10 + [540] * 10 + [500] * 10
    for i, y in enumerate(ys):
        act.update(frame(i * .1, cup_pose("right", y), []))
    assert (act.lift_count, act.place_count) == (1, 0), act.reps
    for i in range(10):
        act.update(frame(4 + i * .1, cup_pose("right", 700), []))
    assert (act.lift_count, act.place_count) == (1, 1), act.reps


def _abduction_hand(mode, delta):
    hand = hand_at("right", (500, 500), 80.0, 60.0)
    pts = hand["pts"]
    length = np.linalg.norm(pts[9] - pts[0])
    across = pts[5] - pts[17]
    across /= np.linalg.norm(across)
    forward = (pts[9] - pts[0]) / length
    pts[2] = pts[5] + forward * 0.4 * length
    pts[4] = pts[2] + forward * 0.4 * length
    if mode == "thumb" and delta > 0:
        pts[4] += across * delta
    elif mode == "index" and delta > 0:
        direction = pts[5] - pts[0]
        direction /= np.linalg.norm(direction)
        pts[6], pts[7], pts[8] = [pts[5] + direction * d for d in (50, 80, 105)]
    pts[3] = (pts[2] + pts[4]) / 2
    hand["palm"] = palm_center(hand["pts"])
    hand["wrist"] = hand["pts"][0].copy()
    return hand


def test_abduction_counts_thumb_and_index_in_any_order():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FingerAbduction(ctx)
    t = 0.0
    for _ in range(6):
        act.update(frame(t, None, [_abduction_hand("thumb", 0.0)])); t += 0.1
    for mode in ("index", "thumb"):
        for delta in (35.0,) * 6 + (0.0,) * 6:
            act.update(frame(t, None, [_abduction_hand(mode, delta)])); t += 0.1
    assert len(act.modes["thumb"]) == 1 and len(act.modes["index"]) == 1, act.modes


def test_abduction_counts_past_old_template_limit():
    act = FingerAbduction(ActionContext("测试", "right", 25, 20))
    t = 0.0
    for delta in (0.0,) * 6 + ((35.0,) * 6 + (0.0,) * 6) * 12:
        act.update(frame(t, None, [_abduction_hand("thumb", delta)]))
        t += .1
    assert len(act.modes["thumb"]) == 12 and not act.modes["index"], act.modes


def test_abduction_world_geometry_is_rotation_and_scale_invariant():
    for side in ("left", "right"):
        act = FingerAbduction(ActionContext("测试", side, 25, 10))
        seq = (0.0,) * 6 + (35.0,) * 8 + (0.0,) * 8
        for i, delta in enumerate(seq):
            hand = _abduction_hand("index", delta)
            hand["side"] = side
            pts = hand["pts"]
            xyz = np.column_stack((pts - pts[0], np.zeros(21))) * .0005
            theta = i * .05
            matrix = np.array([[math.cos(theta), 0, math.sin(theta)],
                               [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
            hand["world"] = xyz @ matrix.T
            hand["pts"] = pts * (1 + .02 * i) + np.array([i * 5, i * 3])
            act.update(frame(i * .1, None, [hand]))
        assert len(act.modes["index"]) == 1 and act.passed == 1, act.modes


def test_abduction_does_not_count_return_during_prediction_or_other_side():
    act = FingerAbduction(ActionContext("测试", "right", 25, 10))
    for i, delta in enumerate((0.0,) * 6 + (35.0,) * 6):
        act.update(frame(i * .1, None, [_abduction_hand("index", delta)]))
    for i in range(4):
        predicted = {**_abduction_hand("index", 0.0), "state": "bridged"}
        other = {**_abduction_hand("index", 35.0), "side": "left"}
        act.update(frame(1.2 + i * .1, None, [predicted, other]))
    assert act.count == 0, act.reps


def test_abduction_whole_hand_open_or_translation_does_not_count():
    act = FingerAbduction(ActionContext("测试", "right", 25, 10))
    for i in range(24):
        hand = _abduction_hand("thumb", 0.0)
        if 6 <= i < 14:
            opened = hand_at("right", (500, 500), 175.0, 175.0)
            for j in range(5, 21):
                hand["pts"][j] = opened["pts"][j]
        hand["pts"] += np.array([i * 10, i * 5])
        act.update(frame(i * .1, None, [hand]))
    assert act.count == 0, act.reps


def test_press_counts_release_cycles_without_leaving_target():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FingerPress(ctx)
    t = 0.0
    for i in range(4):
        base = hand_at("right", (600, 600))
        base["pts"][8] = np.array([600.0, 650.0])
        act.update(frame(t, None, [base])); t += .1
        down = hand_at("right", (600, 600))
        down["pts"][8] = np.array([600.0, 680.0])
        act.update(frame(t, None, [down])); t += .1
        act.update(frame(t, None, [down])); t += .1
        act.update(frame(t, None, [base])); t += .1
    act.update(frame(t, None, [base]))
    assert len(act.targets["a"]) == 4, act.targets



# ---------------- Qt 界面联调（离屏） ----------------
def _window():
    from PySide6.QtWidgets import QApplication

    import main

    app = QApplication.instance() or QApplication([])
    w = main.MainWindow()
    return main, w


def test_ui_flow_count_and_summary():
    _app_module, w = _window()
    w.in_name.setText("张三")
    w.in_side.setCurrentIndex(1)          # 右侧患侧
    w.in_forearm.setValue(25.0)
    w.in_target.setValue(2)
    w.goto_actions()
    assert w.patient["side"] == "right" and w.patient["target_reps"] == 2

    w.start_action(FistOpenClose)
    assert w.stack.currentIndex() == 2
    w.ctx.forearm_px = 300.0
    w.start_running()
    assert w.running

    for t, is_open in [(0.0, True), (0.5, True), (0.9, False), (1.4, False), (2.0, True)]:
        deg = (175.0, 175.0) if is_open else (80.0, 60.0)
        w.on_metrics(frame(t, None, [hand_at("right", (600, 600), *deg), hand_at("left", (400, 600), *deg)]))

    assert w.action.count == 1, w.action.reps
    assert w.action.passed == 1, w.action.reps
    assert w.lb_count.text() == "1" and w.lb_pass.text() == "1"

    w.finish_running()
    assert w.stack.currentIndex() == 3
    assert w.sum_labels["count"].text() == "1"
    assert w.sum_labels["rate"].text() == "100%"


def test_qt_action_template_mapping():
    import main

    thread = main.CaptureThread()
    for cls in main.REGISTRY:
        thread.set_action_key(cls.key)
        assert isinstance(thread.guidance_rois, dict), cls.key


def test_abduction_main_and_server_show_same_counts():
    import server

    _, window = _window()
    window.in_side.setCurrentIndex(1)
    window.goto_actions()
    window.start_action(FingerAbduction)
    window.start_running()
    session = server.TrainingSession({"action": "abduction", "patient": {
        "affected_side": "right", "forearm_cm": 25, "target_reps": 10,
    }})
    seq = (0.0,) * 6 + (35.0,) * 6 + (0.0,) * 6
    for i, delta in enumerate(seq):
        f = frame(i * .1, None, [_abduction_hand("thumb", delta)])
        window.on_metrics(f)
        status = session.update(f)
    assert window.lb_count.text() == "1"
    assert status["thumb_count"] == 1 and status["index_count"] == 0
    window.finish_running()
    assert window.sum_labels["count"].text() == "1"
    assert session.summary()["count"] == 1
    window.close()


def test_stretch_main_server_automatic_direction_without_mode_control():
    import server

    _, window = _window()
    window.in_side.setCurrentIndex(1)
    window.goto_actions()
    window.start_action(FingerStretch)
    window.start_running()
    assert not hasattr(window, "cmb_mode")
    session = server.TrainingSession({"action": "stretch", "mode": "flex", "patient": {
        "affected_side": "right", "forearm_cm": 25, "target_reps": 3,
    }})
    for f in stretch_frames("ext", 175.0, 165.0, hold=1.0):
        window.on_metrics(f)
        status = session.update(f)
    assert status["ext_count"] == len(window.action.modes["ext"]) == 1
    assert status["flex_count"] == 0
    assert session.action.mode == "ext" and window.action.mode == "ext"
    window.close()


def test_cup_main_server_show_same_pose_only_events():
    import server

    _, window = _window()
    window.in_side.setCurrentIndex(1)
    window.goto_actions()
    window.start_action(CupTransfer)
    window.start_running()
    session = server.TrainingSession({"action": "cup", "patient": {
        "affected_side": "right", "forearm_cm": 25, "target_reps": 10,
    }})
    for i, y in enumerate([700] * 10 + [590] * 10 + [700] * 10):
        f = frame(i * .1, cup_pose("right", y), [])
        window.on_metrics(f)
        status = session.update(f)
    assert window.lb_count.text() == "2"
    assert (status["lift_count"], status["place_count"]) == (1, 1)
    assert (window.action.lift_count, window.action.place_count) == (1, 1)
    window.finish_running()
    assert window.sum_labels["detail"].text() == "端起 1 / 放下 1"
    assert session.summary()["detail"] == "端起 1 / 放下 1"
    window.close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL %s -> %s" % (fn.__name__, e))
    print("\n%d/%d 通过" % (len(tests) - failed, len(tests)))
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
