"""可运行自检：几何/角度 + 三个动作状态机 + Qt 界面联调（合成关键点，不需要摄像头）。

运行：/opt/anaconda3/envs/yolo26-pose/bin/python test_rehab.py
"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 无摄像头/无显示器也能跑界面联调

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
    assert ctx.elbow_stale and act.phase in ("回程", "outbound"), (ctx.elbow_stale, act.phase)


# ---------------- 动作3 牵伸手指 ----------------
def hand_grabbing(side, target_tip_center, mcp_deg=175.0, pip_deg=175.0):
    """健侧手：让它的指尖落在指定位置（模拟捏住患侧指尖）。"""
    h = hand_at(side, (0.0, 0.0), mcp_deg, pip_deg)
    shift = np.asarray(target_tip_center, float) - fingertips(h["pts"]).mean(axis=0)
    pts = h["pts"] + shift
    return {
        "side": side,
        "pts": pts,
        "angles": finger_angles(pts),
        "palm": palm_center(pts),
        "wrist": pts[0].copy(),
    }


def stretch_frames(mode, deg_from, deg_to, hold=3.0):
    out, t = [], 0.0
    n = int(hold / 0.1)
    for i in range(n + 1):
        deg = deg_from + (deg_to - deg_from) * i / max(n, 1)
        aff = hand_at("right", (600.0, 620.0), deg, deg)
        heal = hand_grabbing("left", fingertips(aff["pts"]).mean(axis=0))
        out.append(frame(t, None, [aff, heal]))
        t += 0.1
    aff = hand_at("right", (600.0, 620.0), deg_to, deg_to)
    out.append(frame(t, None, [aff, hand_grabbing("left", (900.0, 300.0))]))
    return out


def test_stretch_flex_counts_press():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FingerStretch(ctx, mode="flex")
    for f in stretch_frames("flex", 175.0, 120.0):   # 真实掌屈牵伸幅度
        act.update(f)
    assert act.count == 1, (act.count, act.reps)
    assert act.passed == 1, act.reps
    assert act.modes["flex"] and not act.modes["ext"]


def test_stretch_ext_needs_only_small_change():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FingerStretch(ctx, mode="ext")
    for f in stretch_frames("ext", 150.0, 156.0, hold=3.0):
        act.update(f)
    assert act.count == 1, act.reps
    assert act.passed == 1, act.reps


def test_stretch_too_short_not_counted():
    ctx = ActionContext("测试", "right", 25.0, 10)
    ctx.forearm_px = 300.0
    act = FingerStretch(ctx, mode="flex")
    for f in stretch_frames("flex", 175.0, 120.0, hold=0.4):
        act.update(f)
    assert act.count == 0, act.reps


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


def _abduction_hand(mode, delta):
    hand = hand_at("right", (500, 500))
    if mode == "thumb":
        hand["pts"][4] += np.array([-delta, 0.0])
    else:
        hand["pts"][8] += np.array([-delta, 0.0])
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
        for delta in (35.0, 35.0, 35.0, 0.0, 0.0, 0.0):
            act.update(frame(t, None, [_abduction_hand(mode, delta)])); t += 0.1
    assert len(act.modes["thumb"]) == 1 and len(act.modes["index"]) == 1, act.modes


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

    import qt_app

    app = QApplication.instance() or QApplication([])
    w = qt_app.MainWindow()
    return qt_app, w


def test_ui_flow_count_and_summary():
    qt_app, w = _window()
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
    import qt_app

    thread = qt_app.CaptureThread()
    for cls in qt_app.REGISTRY:
        thread.set_action_key(cls.key)
        assert isinstance(thread.guidance_rois, dict), cls.key


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
