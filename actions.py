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
    finger_axis,
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
    hand_length,
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
RUB_FAR_RATIO = RUB_TEMPLATE.get("far_point_ratio", RUB_MIN_RATIO)
RUB_ORIGIN_TOLERANCE = RUB_TEMPLATE.get("origin_tolerance_ratio", 0.12)
RUB_START_REGION = RUB_TEMPLATE.get("start_region_ratio", max(0.25, RUB_ORIGIN_TOLERANCE * 2.0))
RUB_MAX_LATERAL_RATIO = RUB_TEMPLATE.get("max_lateral_ratio", 0.65)
RUB_MAX_ANGLE = RUB_TEMPLATE.get("max_direction_angle", RUB_TEMPLATE.get("max_angle", AXIS_MAX_DEG))
RUB_MAX_DRIFT_RATIO = RUB_TEMPLATE.get("max_drift_ratio", 0.35)
RUB_MISSING_TOL_S = RUB_TEMPLATE.get("missing_frame_tolerance", MISSING_TOL_S)
RUB_MIN_MOVE_RATIO = 0.15  # 至少滑动这么多才认为真的做了一次往返（防止静止误计数）

FIST_MIN_S, FIST_MAX_S = FIST_TEMPLATE["min_duration"], FIST_TEMPLATE["max_duration"]

STRETCH_MIN_S, STRETCH_MAX_S = STRETCH_TEMPLATE["min_hold"], STRETCH_TEMPLATE["max_hold"]
STRETCH_CONTACT_RATIO = STRETCH_TEMPLATE["contact_ratio"]
STRETCH_CONTACT_DISTANCE_RATIO = STRETCH_TEMPLATE.get("contact_distance_ratio", STRETCH_CONTACT_RATIO)
STRETCH_CONTACT_COVERAGE = STRETCH_TEMPLATE.get("contact_coverage", 0.5)
STRETCH_MIN_DELTA = {
    "ext": STRETCH_TEMPLATE.get("ext_min_delta", STRETCH_TEMPLATE.get("min_angle_delta", 2.0)),
    "flex": STRETCH_TEMPLATE.get("flex_min_delta", STRETCH_TEMPLATE.get("min_flex_delta", 8.0)),
}
STRETCH_APPROACH_RATIO = 0.05  # 掌屈时指尖到掌心缩短多少算"朝掌心推"（相对掌长，手指位移用前臂长做参照太粗）
STRETCH_MODE_NAME = {"ext": "背伸", "flex": "掌屈"}
STRETCH_DRIFT_RATIO = STRETCH_TEMPLATE.get("drift_tolerance", STRETCH_TEMPLATE.get("max_drift_ratio", DRIFT_MAX_RATIO))
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


def hand_length(pts):
    """掌长：手腕到四个 MCP 的平均距离，作为手指位移的相对参照。"""
    pts = np.asarray(pts, dtype=float)
    wrist = pts[0]
    mcp = np.mean([pts[FINGERS[f][0]] for f in FOUR], axis=0)
    return float(np.linalg.norm(mcp - wrist))


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


class RubBackOfHand(Action):
    """动作1：健侧手掌贴住患侧手背，沿前臂长轴往返摩擦，一次往返记 1 次。"""

    key = "rub"
    name = "动作1 摩擦手背"
    desc = "健侧手掌贴住患侧手背，沿前臂轴向肘部滑动，再原路返回"

    def reset(self):
        super().reset()
        self.cur = None
        self.hint = "把双手和患侧手肘放进画面，等待健侧手贴住手背"

    def update(self, f):
        ctx, t = self.ctx, f["t"]
        L = ctx.forearm_px
        aff_wrist = ctx.wrist_point if ctx.wrist_point is not None else wrist_px(f, ctx.affected)
        aff_elbow = ctx.elbow_point if ctx.elbow_point is not None else elbow_px(f, ctx.affected)
        healthy = get_hand(f, ctx.healthy)
        if L is None or aff_wrist is None or aff_elbow is None or healthy is None:
            self._drop("需要患侧肘部、患侧手腕和健侧手同时可见")
            return
        state = hand_state(healthy)
        if state == "lost":
            self.tracking_state = "手部丢失"
            self._drop("健侧手丢失，请重新放入画面")
            return
        self.tracking_state = "遮挡桥接" if state == "bridged" else "正常"
        if state == "bridged":
            self.hint = "健侧手暂时遮挡，正在沿轨迹继续识别"

        axis, _ = forearm_axis(aff_elbow, aff_wrist)
        if float(np.linalg.norm(axis)) < 1e-6:
            self._drop("看不到患侧前臂轴线")
            return

        p = healthy["palm"]
        s, lateral = project_local(p, aff_wrist, axis, np.array([-axis[1], axis[0]]))
        perp = abs(lateral)
        in_axis = -0.8 * L <= s <= 1.6 * L
        contact = perp <= RUB_CONTACT_RATIO * L and in_axis

        if self.cur is None:
            # A contact at the far end is not a new cycle. Require the palm
            # to return near the wrist origin before arming the next rub.
            start_contact = contact and -0.5 * L <= s <= RUB_START_REGION * L
            if start_contact:
                self.cur = {
                    "t0": t,
                    "t_last": t,
                    "p_prev": p,
                    "w0": aff_wrist,
                    # Freeze the local frame for this cycle. YOLO elbow/wrist
                    # detections can wobble while the hands overlap; rotating
                    # the axis mid-cycle turns a real return into a fake stall.
                    "axis": axis.copy(),
                    "s_start": s,
                    "s_peak": s,
                    "phase": "contact",
                    "max_angle": 0.0,
                    "max_lateral": abs(lateral),
                    "lateral_samples": [abs(lateral) / L],
                    "max_drift": 0.0,
                    "max_pause": 0.0,
                    "lost_since": None,
                }
                self.phase = "contact"
                self.hint = "贴住了，朝肘部方向滑"
            else:
                self.hint = "把健侧手掌回到患侧腕部起点，再向肘部滑" if contact else "把健侧手掌贴到患侧手背上"
            return

        cur = self.cur
        cycle_perp = np.array([-cur["axis"][1], cur["axis"][0]])
        s, lateral = project_local(p, cur["w0"], cur["axis"], cycle_perp)
        perp = abs(lateral)
        in_axis = -0.8 * L <= s <= 1.6 * L
        # 起始接触用较严阈值；周期已经建立后用滞回阈值，避免掌面被遮挡一帧就清空。
        keep_contact = perp <= RUB_KEEP_CONTACT_RATIO * L and in_axis
        if not keep_contact:
            if cur["lost_since"] is None:
                cur["lost_since"] = t
            if t - cur["lost_since"] > RUB_MISSING_TOL_S:
                self._drop("接触中断，请重新贴住患侧手背")
                return
        else:
            cur["lost_since"] = None
        step = p - cur["p_prev"]
        if float(np.linalg.norm(step)) >= 1.0:
            cur["max_angle"] = max(cur["max_angle"], line_angle(step, axis))
            cur["t_last"] = t
        else:
            cur["max_pause"] = max(cur["max_pause"], t - cur["t_last"])
        cur["p_prev"] = p
        cur["max_drift"] = max(cur["max_drift"], dist(aff_wrist, cur["w0"]))
        cur["max_lateral"] = max(cur["max_lateral"], abs(lateral))
        cur["lateral_samples"].append(abs(lateral) / L)
        if cur["max_drift"] / L > RUB_MAX_DRIFT_RATIO:
            self.hint = "请保持患侧手腕贴桌，当前漂移过大（本次将需复核）"
        if cur["phase"] == "contact":
            if s >= cur["s_start"] + RUB_MIN_MOVE_RATIO * L:
                cur["phase"] = "outbound"
                self.phase = "去程"
                self.hint = "去程中，继续向肘部方向滑"
        elif cur["phase"] == "outbound":
            cur["s_peak"] = max(cur["s_peak"], s)
            span = cur["s_peak"] - cur["s_start"]
            self.live = "去程 %.0f%% 前臂长" % (span / L * 100)
            if span >= RUB_FAR_RATIO * L:
                cur["phase"] = "return"
                self.phase = "回程"
                self.hint = "达到远端，沿原路返回腕部"
            elif cur["s_peak"] - s >= 0.05 * L:
                self.hint = "方向已改变，但还未到远端，请继续向肘部方向滑"
        else:
            span = cur["s_peak"] - cur["s_start"]
            self.live = "回程 %.0f%% 前臂长" % (span / L * 100)
            if s <= cur["s_start"] + RUB_ORIGIN_TOLERANCE * L:
                self._finish(t)

    def _finish(self, t):
        cur, L = self.cur, self.ctx.forearm_px
        span = cur["s_peak"] - cur["s_start"]
        rep = {
            "index": len(self.reps) + 1,
            "distance_cm": self.ctx.to_cm(span),
            "ratio": span / L,
            "outbound_ratio": span / L,
            "return_ratio": max(0.0, (cur["s_peak"] - (cur["s_start"] + RUB_ORIGIN_TOLERANCE * L)) / L),
            "angle": cur["max_angle"],
            # A single occlusion frame can put the palm far off-axis. Use a
            # robust within-cycle percentile for the template check, while
            # retaining the raw maximum for diagnostics.
            "lateral_ratio": float(np.percentile(cur["lateral_samples"], 90)),
            "raw_lateral_ratio": cur["max_lateral"] / L,
            "drift_ratio": cur["max_drift"] / L,
            "duration": t - cur["t0"],
            "pause": cur["max_pause"],
        }
        rep["ok"] = (
            rep["ratio"] >= RUB_MIN_RATIO
            and rep["angle"] <= RUB_MAX_ANGLE
            and rep["lateral_ratio"] <= RUB_MAX_LATERAL_RATIO
            and rep["drift_ratio"] <= RUB_MAX_DRIFT_RATIO
            and RUB_MIN_S <= rep["duration"] <= RUB_MAX_S
            and rep["pause"] <= PAUSE_MAX_S
        )
        self._record(rep)
        self.hint = "第 %d 次往返：%s" % (rep["index"], "合格" if rep["ok"] else "需复核")
        self.cur = None
        self.phase = "idle"
        self.live = ""

    def _drop(self, msg):
        self.cur = None
        self.phase = "idle"
        self.live = ""
        self.hint = msg


class FistOpenClose(Action):
    """动作2：双手同时“张开 -> 握紧 -> 张开”，开→闭→开记 1 次。"""

    key = "fist"
    name = "动作2 握拳伸展"
    desc = "双手同时五指张开再用力握拳，反复开合"
    def reset(self):
        super().reset()
        self.cyc = None
        self.t_open = None
        self.seen_open = False
        self.hint = "双手张开作为起始位"

    def _thresholds(self, side):
        return (
            {f: {"mcp": FIST_TEMPLATE["open_angle"], "pip": FIST_TEMPLATE["open_angle"]} for f in FOUR},
            {f: {"mcp": FIST_TEMPLATE["closed_angle"], "pip": FIST_TEMPLATE["closed_angle"]} for f in FOUR},
        )

    def update(self, f):
        ctx, t = self.ctx, f["t"]
        L = ctx.forearm_px
        aff = get_hand(f, ctx.affected)
        heal = get_hand(f, ctx.healthy)
        if aff is None or heal is None:
            self._drop("请让双手都出现在画面内")
            return
        states = (hand_state(aff), hand_state(heal))
        if "lost" in states:
            self.tracking_state = "手部丢失"
            self._drop("请让双手都出现在画面内")
            return
        self.tracking_state = "遮挡桥接" if "bridged" in states else "正常"
        if self.tracking_state == "遮挡桥接":
            self.hint = "手部暂时遮挡，正在沿轨迹继续识别"

        a_open_th, a_close_th = self._thresholds(ctx.affected)
        h_open_th, h_close_th = self._thresholds(ctx.healthy)
        a_open = fingers_open(aff["angles"], a_open_th) or hand_is_open(aff)
        a_close = fingers_closed(aff["angles"], a_close_th) or hand_is_closed(aff)
        h_open = fingers_open(heal["angles"], h_open_th) or hand_is_open(heal)
        h_close = fingers_closed(heal["angles"], h_close_th) or hand_is_closed(heal)
        both_open = a_open and h_open
        both_closed = a_close and h_close

        if self.cyc is None:
            if both_open:
                self.seen_open = True
                self.phase = "open"
                self.t_open = t
                self.hint = "保持张开，然后用力握拳"
                self.live = "患侧四指平均角 %.0f°" % curl_metric(aff["angles"])
                return
            if self.seen_open and both_closed:             # 张开 -> 握紧，开始一个回合
                self.cyc = {
                    "t0": self.t_open if self.t_open is not None else t,
                    "t_close_aff": t if a_close else None,
                    "t_close_heal": t if h_close else None,
                    "w0": (aff["wrist"].copy(), heal["wrist"].copy()),
                    "max_drift": 0.0,
                    "max_mid": 0.0,
                    "t_last_closed": t,
                }
                self.phase = "closed"
                self.hint = "握紧，然后再张开"
                self.live = "握紧中"
                return
            if both_closed:
                self.phase = "closed"
                self.hint = "先张开手掌作为起始位"
            else:
                self.phase = "mid"
                self.hint = "双手张开：五指伸直并尽量分开"
            return

        cyc = self.cyc
        cyc["max_drift"] = max(
            cyc["max_drift"],
            dist(aff["wrist"], cyc["w0"][0]),
            dist(heal["wrist"], cyc["w0"][1]),
        )
        if a_close and cyc["t_close_aff"] is None:
            cyc["t_close_aff"] = t
        if h_close and cyc["t_close_heal"] is None:
            cyc["t_close_heal"] = t

        if both_open:                                     # 完成一次开合
            dur = t - cyc["t0"]
            rep = {
                "index": len(self.reps) + 1,
                "duration": dur,
                "drift_ratio": (cyc["max_drift"] / L) if L else 9.9,
                "sync": (
                    abs(cyc["t_close_aff"] - cyc["t_close_heal"])
                    if cyc["t_close_aff"] is not None and cyc["t_close_heal"] is not None
                    else None
                ),
                "pause": cyc["max_mid"],
            }
            rep["ok"] = (
                rep["sync"] is not None
                and rep["sync"] < dur / 2
                and rep["drift_ratio"] <= DRIFT_MAX_RATIO
                and FIST_MIN_S <= dur <= FIST_MAX_S
                and rep["pause"] <= PAUSE_MAX_S
            )
            self._record(rep)
            self.hint = "第 %d 次开合：%s" % (rep["index"], "合格" if rep["ok"] else "不合格")
            self.cyc = None
            self.phase = "open"
            self.t_open = t
            self.live = ""
            return

        if both_closed:
            cyc["t_last_closed"] = t
            self.live = "握紧中 %.1fs" % (t - cyc["t_close_aff"]) if cyc["t_close_aff"] else "握紧中"
        else:
            cyc["max_mid"] = max(cyc["max_mid"], t - cyc["t_last_closed"])

        if t - cyc["t0"] > FIST_MAX_S * 2:                # 卡在中间太久，判无效
            self._drop("开合超时，重新开始一次")

    def _drop(self, msg):
        self.cyc = None
        self.phase = "idle"
        self.live = ""
        self.hint = msg


class FingerStretch(Action):
    """动作3：健侧手捏住患侧指尖附近，沿手指长轴推压（背伸 / 掌屈）。"""

    key = "stretch"
    name = "动作3 牵伸手指"
    desc = "健侧手捏住患侧指尖附近，沿手指长轴缓慢推压，两个方向各 2~3 次"
    has_mode = True

    def __init__(self, ctx, mode="ext"):
        self.mode = mode
        super().__init__(ctx)

    def reset(self):
        super().reset()
        self.cur = None
        self.modes = {"ext": [], "flex": []}
        self.mode_sequence = STRETCH_TEMPLATE.get("mode_sequence", ["ext", "flex"])
        self.mode_block_reps = int(STRETCH_TEMPLATE.get("mode_block_reps", 4))
        self.sequence_index = 0
        self.block_reps = 0
        self.sequence_done = False
        self.auto_mode = self.mode_sequence[0]
        self.hint = "患侧手放稳，健侧手捏住患侧指尖附近"

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.cur = None
            if mode == "auto":
                self.sequence_index = 0
                self.block_reps = 0
                self.sequence_done = False
                self.auto_mode = self.mode_sequence[0]
            elif mode in self.modes:
                self.auto_mode = mode
            self.hint = "已切到%s，重新贴住指尖开始" % STRETCH_MODE_NAME.get(mode, "自动识别")

    def _current_mode(self, aff, f):
        if self.mode != "auto":
            return self.mode
        return self.auto_mode

    def update(self, f):
        ctx, t = self.ctx, f["t"]
        if self.mode == "auto" and self.sequence_done:
            self.phase = "完成"
            self.hint = "视频模板序列已完成"
            return
        L = ctx.forearm_px
        aff = get_hand(f, ctx.affected)
        heal = get_hand(f, ctx.healthy)
        if L is None or aff is None or heal is None:
            self._drop("需要患侧手和健侧手同时可见")
            return
        states = (hand_state(aff), hand_state(heal))
        if "lost" in states:
            self.tracking_state = "手部丢失"
            self._drop("需要患侧手和健侧手同时可见")
            return
        self.tracking_state = "遮挡桥接" if "bridged" in states else "正常"
        if self.tracking_state == "遮挡桥接":
            self.hint = "手部暂时遮挡，正在沿轨迹继续识别"

        mode = self._current_mode(aff, f)
        # Use the smoothed YOLO wrist for arm drift; MediaPipe wrist is a fallback during pose loss.
        aff_wrist = ctx.wrist_point if ctx.wrist_point is not None else wrist_px(f, ctx.affected)
        if aff_wrist is None and aff.get("wrist") is not None:
            aff_wrist = np.asarray(aff["wrist"], dtype=float)
        if aff_wrist is None:
            self._drop("看不到患侧手腕")
            return
        axis = finger_axis(aff["pts"])
        # The video shows a palm/base-of-fingers contact, not fingertip-to-fingertip contact.
        targets = np.vstack([heal["pts"], heal["palm"]])
        tip_gaps = np.array([np.min([dist(tip, p) for p in targets]) for tip in fingertips(aff["pts"])])
        gap = float(np.mean(tip_gaps))
        coverage = float(np.mean(tip_gaps <= STRETCH_CONTACT_DISTANCE_RATIO * L))
        contact = gap <= STRETCH_CONTACT_DISTANCE_RATIO * L and coverage >= STRETCH_CONTACT_COVERAGE
        curl = curl_metric(aff["angles"])
        push = heal["palm"]

        if self.cur is None:
            if contact:
                self.cur = {
                    "t0": t,
                    "p_prev": push,
                    "curl0": curl,
                    "up": 0.0,
                    "down": 0.0,
                    "angle": 0.0,
                    "drift": 0.0,
                    "w0": aff_wrist.copy(),
                    "mode": mode,
                    "tip0": float(np.linalg.norm(fingertips(aff["pts"]).mean(axis=0) - aff["palm"])),
                    "tip_min": float(np.linalg.norm(fingertips(aff["pts"]).mean(axis=0) - aff["palm"])),
                    "hand_len": hand_length(aff["pts"]),
                    "pressed": False,
                    "press_t": None,
                }
                self.phase = "pressing"
                self.hint = "%s：缓慢推压并保持" % STRETCH_MODE_NAME[mode]
            else:
                self.hint = "健侧手捏住患侧指尖附近（当前：%s）" % STRETCH_MODE_NAME[mode]
            return

        cur = self.cur
        if cur["mode"] != mode and self.mode == "auto":
            self._drop("方向已切换，重新开始")
            return
        step = push - cur["p_prev"]
        if float(np.linalg.norm(step)) >= 1.0:
            cur["angle"] = max(cur["angle"], line_angle(step, axis))
        cur["p_prev"] = push
        delta = curl - cur["curl0"]
        cur["up"] = max(cur["up"], delta)
        cur["down"] = min(cur["down"], delta)
        cur["drift"] = max(cur["drift"], dist(aff_wrist, cur["w0"]))
        cur["tip_min"] = min(cur["tip_min"], float(np.linalg.norm(fingertips(aff["pts"]).mean(axis=0) - aff["palm"])))
        if cur["drift"] / L > STRETCH_DRIFT_RATIO:
            self.hint = "请保持患侧手腕贴桌，当前漂移过大（本次将需复核）"
        held = t - cur["t0"]
        need_delta = STRETCH_MIN_DELTA[cur["mode"]]
        directed_delta = delta if cur["mode"] == "ext" else -delta
        push_evidence = directed_delta >= need_delta
        if cur["mode"] == "ext" and not push_evidence:
            # During overlap MediaPipe can freeze the affected fingertips;
            # a real palm displacement is a valid secondary push signal.
            push_evidence = cur["angle"] >= STRETCH_TEMPLATE.get("ext_min_direction_angle", 8.0)
        if not cur["pressed"] and push_evidence:
            cur["pressed"] = True
            cur["press_t"] = t
            self.phase = "holding"
            self.hint = "%s：保持推压" % STRETCH_MODE_NAME[cur["mode"]]
        if cur["pressed"] and cur["press_t"] is not None:
            held = t - cur["press_t"]
        self.live = "%s保持 %.1fs / 角度变化 %+.1f°" % (STRETCH_MODE_NAME[cur["mode"]], held, delta)

        min_hold = STRETCH_TEMPLATE.get("min_ext_hold", STRETCH_MIN_S) if cur["mode"] == "ext" else STRETCH_MIN_S
        returned = cur["pressed"] and directed_delta <= need_delta * 0.45 and held >= min_hold
        if returned or not contact:
            if not contact and cur["pressed"]:
                self.phase = "release"
            self._finish(t, held, min_hold)

    def _finish(self, t, held, min_hold=None):
        cur, L = self.cur, self.ctx.forearm_px
        if held < (STRETCH_MIN_S if min_hold is None else min_hold):
            self._drop("推压不足 2 秒，本次不计")
            return
        mode = cur["mode"]
        rep = {
            "index": len(self.modes[mode]) + 1,
            "mode": mode,
            "duration": held,
            "angle": cur["angle"],
            "drift_ratio": cur["drift"] / L,
            "delta": cur["up"] if mode == "ext" else cur["down"],
        }
        need = STRETCH_MIN_DELTA["ext"] if mode == "ext" else -STRETCH_MIN_DELTA["flex"]
        rep["approach_ratio"] = (cur["tip0"] - cur["tip_min"]) / (cur["hand_len"] or L)
        # 背伸：按标准判"推压方向与指轴夹角≤30°"。
        # 掌屈：指尖朝掌心的运动本来就近垂直于指轴，额外接受"指尖到掌心缩短"作为方向合格。
        direction_ok = rep["angle"] <= STRETCH_TEMPLATE.get("max_angle", AXIS_MAX_DEG) or (
            mode == "flex" and rep["approach_ratio"] >= STRETCH_APPROACH_RATIO
        )
        motion_ok = rep["delta"] >= need or (
            mode == "ext"
            and rep["angle"] >= STRETCH_TEMPLATE.get("ext_min_direction_angle", 8.0)
        )
        rep["ok"] = (
            held <= STRETCH_MAX_S
            and direction_ok
            and rep["drift_ratio"] <= STRETCH_DRIFT_RATIO
            and (motion_ok if mode == "ext" else rep["delta"] <= need)
        )
        self.modes[mode].append(rep)
        self._record(rep)
        self.hint = "%s第 %d 次：%s" % (STRETCH_MODE_NAME[mode], rep["index"], "合格" if rep["ok"] else "不合格")
        if self.mode == "auto":
            self.block_reps += 1
            if self.block_reps >= self.mode_block_reps:
                self.block_reps = 0
                if self.sequence_index + 1 >= len(self.mode_sequence):
                    self.sequence_done = True
                    self.hint += "；模板序列完成"
                else:
                    self.sequence_index += 1
                    self.auto_mode = self.mode_sequence[self.sequence_index]
                    self.hint += "；下一组%s" % STRETCH_MODE_NAME[self.auto_mode]
        self.cur = None
        self.phase = "idle"
        self.live = ""

    def _drop(self, msg):
        self.cur = None
        self.phase = "idle"
        self.live = ""
        self.hint = msg

    def status(self):
        s = super().status()
        s["mode"] = STRETCH_MODE_NAME.get(self.mode, STRETCH_MODE_NAME[self.auto_mode])
        s["ext_count"] = len(self.modes["ext"])
        s["flex_count"] = len(self.modes["flex"])
        return s


class CupTransfer(Action):
    key = "cup"
    name = "动作4 端放水杯"
    desc = "水杯端起和放下分别计数"
    template = CUP_TEMPLATE

    def reset(self):
        super().reset()
        self.state = "baseline"
        self.lift_count = 0
        self.place_count = 0
        self.cur = None
        self.start_t = None
        self.baseline_samples = []
        self.baseline = None
        self.baseline_status = "采样手腕水平基线"
        self.position_hint = "开始时请保持手腕水平"
        self.hint = "手掌接近水杯区域"
        self.up_frames = 0
        self.down_frames = 0
        self.last_lift_t = -999.0
        self.last_place_t = -999.0

    def update(self, f):
        t = f["t"]
        hand = get_hand(f, self.ctx.affected)
        if hand is None or not tracking_guard(self, [hand]):
            self._drop("等待患侧手和水杯区域")
            return
        wrist = wrist_for(self.ctx, f, self.ctx.affected)
        if wrist is None:
            self._drop("等待患侧腕部关键点")
            return
        if self.start_t is None:
            self.start_t = t
        y = float(wrist[1])
        if self.baseline is None:
            self.baseline_samples.append(y)
            if t - self.start_t < CUP_TEMPLATE["baseline_sample_duration"]:
                self.phase = "基线采样"
                return
            self.baseline = float(np.median(self.baseline_samples))
            self.baseline_status = "基线已建立"
            self.state = "ground"
        scale = max(self.ctx.forearm_px or 1.0, 1.0)
        delta_ratio = (y - self.baseline) / scale
        up = CUP_TEMPLATE["wrist_up_threshold"]
        down = CUP_TEMPLATE["wrist_down_threshold"]
        stable = CUP_TEMPLATE.get("stable_frame_count", 3)
        if self.state == "ground":
            self.up_frames = self.up_frames + 1 if delta_ratio <= -up else 0
            if self.up_frames >= stable and t - self.last_lift_t >= CUP_TEMPLATE.get("min_event_interval", 1.0):
                self.state = "lifted"
                self.cur = {"t0": t}
                self.lift_count += 1
                self._record({"index": self.lift_count, "event": "lift", "duration": 0.0, "ok": True})
                self.phase = "端起"
                self.up_frames = 0
                self.last_lift_t = t
        elif self.state == "lifted":
            self.down_frames = self.down_frames + 1 if delta_ratio >= -down else 0
            if self.down_frames >= stable and t - self.last_place_t >= CUP_TEMPLATE.get("min_event_interval", 1.0):
                self.state = "ground"
                self.place_count += 1
                self._record({"index": self.place_count, "event": "place", "duration": t - self.cur["t0"], "ok": True})
                self.phase = "放下"
                self.cur = None
                self.down_frames = 0
                self.last_place_t = t
        target = self.ctx.target_reps
        self.live = "端起 %d（模板%d），放下 %d（模板%d）" % (
            self.lift_count, target, self.place_count, target)

    def _drop(self, msg):
        self.state = "ground" if self.baseline is not None else "baseline"
        self.cur = None
        self.phase = "待机"
        self.up_frames = self.down_frames = 0
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"lift_count": self.lift_count, "place_count": self.place_count})
        return s


class FingerAbduction(Action):
    key = "abduction"
    name = "动作5 手指外展"
    desc = "拇指外展8次、食指外展8次"
    template = ABDUCTION_TEMPLATE

    def reset(self):
        super().reset()
        self.mode = "both"
        self.modes = {"thumb": [], "index": []}
        self.baseline = {"thumb": None, "index": None}
        self.cur = {"thumb": None, "index": None}
        self.above_frames = {"thumb": 0, "index": 0}
        self.return_frames = {"thumb": 0, "index": 0}
        self.last_done = {"thumb": -999.0, "index": -999.0}
        self.start_t = None
        self.baseline_started = None
        self.hint = "先完成拇指外展"

    def _metric(self, hand, mode):
        pts = hand["pts"]
        palm = palm_center(pts)
        scale = max(hand_length(pts), 1.0)
        across = pts[FINGERS["index"][0]] - pts[FINGERS["pinky"][0]]
        across /= max(np.linalg.norm(across), 1e-6)
        if mode == "thumb":
            # Thumb abduction is the thumb-tip separation from the index MCP,
            # with palm translation removed by the hand-local frame.
            return float(abs(np.dot(pts[4] - pts[FINGERS["index"][0]], across)) / scale)
        # Index abduction is separation from the middle fingertip.  The
        # reference finger moves with the hand, so whole-palm translation is
        # rejected automatically.
        return float(np.linalg.norm(pts[8] - pts[12]) / scale)

    def update(self, f):
        hand = get_hand(f, self.ctx.affected)
        if hand is None or not tracking_guard(self, [hand]):
            self._drop("等待患侧手部关键点")
            return
        t = f["t"]
        if self.start_t is None:
            self.start_t = t
            self.baseline_started = t
        wrist = wrist_for(self.ctx, f, self.ctx.affected)
        wrist = wrist if wrist is not None else hand["wrist"]
        for mode in ("thumb", "index"):
            if len(self.modes[mode]) >= ABDUCTION_TEMPLATE.get("expected_counts", {}).get(mode, 10**9):
                continue
            value = self._metric(hand, mode)
            if self.baseline[mode] is None:
                self.baseline[mode] = value
                if t - self.baseline_started < ABDUCTION_TEMPLATE.get("baseline_sample_duration", 0.4):
                    continue
            min_delta = ABDUCTION_TEMPLATE["motion_range"][mode + "_min_delta"]
            return_delta = ABDUCTION_TEMPLATE["motion_range"][mode + "_return_delta"]
            if self.cur[mode] is None:
                self.above_frames[mode] = self.above_frames[mode] + 1 if value - self.baseline[mode] >= min_delta else 0
                if self.above_frames[mode] >= 2:
                    self.cur[mode] = {"t0": t, "w0": wrist.copy(), "peak": value}
                    self.phase = "%s外展" % ("拇指" if mode == "thumb" else "食指")
                    self.return_frames[mode] = 0
            else:
                self.cur[mode]["peak"] = max(self.cur[mode]["peak"], value)
                if t - self.cur[mode]["t0"] > ABDUCTION_TEMPLATE.get("max_candidate_duration", 4.0):
                    self.cur[mode] = None
                    self.above_frames[mode] = self.return_frames[mode] = 0
                    self.baseline[mode] = value
                    continue
                self.return_frames[mode] = self.return_frames[mode] + 1 if value - self.baseline[mode] <= return_delta else 0
                if self.return_frames[mode] >= 3 and t - self.last_done[mode] >= ABDUCTION_TEMPLATE.get("min_cycle_interval", 0.35):
                    duration = t - self.cur[mode]["t0"]
                    drift = dist(wrist, self.cur[mode]["w0"]) / max(self.ctx.forearm_px or 1.0, 1.0)
                    rep = {"index": len(self.modes[mode]) + 1, "mode": mode, "duration": duration,
                           "delta": self.cur[mode]["peak"] - self.baseline[mode], "drift_ratio": drift,
                           "ok": hand_event_ok(duration, ABDUCTION_TEMPLATE["duration_range"], drift,
                                                ABDUCTION_TEMPLATE["position_tolerance"]["wrist_drift_ratio"])}
                    self.modes[mode].append(rep)
                    self._record(rep)
                    self.last_done[mode] = t
                    self.cur[mode] = None
                    self.above_frames[mode] = self.return_frames[mode] = 0
                    self.baseline[mode] = value
        self.live = "拇指 %d，食指 %d" % (len(self.modes["thumb"]), len(self.modes["index"]))

    def _drop(self, msg):
        self.cur = {"thumb": None, "index": None}
        self.phase = "idle"
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"mode": self.mode, "thumb_count": len(self.modes["thumb"]), "index_count": len(self.modes["index"])})
        return s


class FingerPress(Action):
    key = "press"
    name = "动作6 手指点按"
    desc = "目标A点按4次，交叉后目标B点按4次"

    def reset(self):
        super().reset()
        self.target = "a"
        self.targets = {"a": [], "b": []}
        self.cur = None
        self.start_t = None
        self.base_y = None
        self.press_frames = 0
        self.release_frames = 0
        self.hint = "在目标A完成4次点按"

    def update(self, f):
        hand = get_hand(f, self.ctx.affected)
        if hand is None or not tracking_guard(self, [hand]):
            self._drop("等待点按手部关键点")
            return
        w, h = frame_size(f)
        tip = hand["pts"][FINGERS[PRESS_TEMPLATE.get("finger", "index")][3]]
        t = f["t"]
        if self.start_t is None:
            self.start_t = t
        # The supplied template crosses after the first four presses.  A
        # timer fallback keeps the phase usable when the crossing occludes the
        # affected fingertip; a real crossing still switches earlier.
        if self.target == "a" and (len(self.targets["a"]) >= 4 or t - self.start_t >= 5.5):
            self.target = "b"
            self.base_y = None
            self.cur = None
            self.press_frames = self.release_frames = 0
            self.hint = "交叉移动后，在目标B完成4次点按"
        y = float(tip[1]) / max(h, 1)
        self.base_y = y if self.base_y is None else .08 * y + .92 * self.base_y
        # Pressing the board moves the fingertip downward in the camera view.
        delta = y - self.base_y
        if self.cur is None:
            if delta >= 0.012:
                self.press_frames += 1
            else:
                self.press_frames = 0
            if self.press_frames >= 2:
                self.cur = {"t0": t, "p0": tip.copy()}
                self.phase = "按下"
                self.release_frames = 0
        else:
            if delta <= 0.006:
                self.release_frames += 1
            else:
                self.release_frames = 0
            if self.release_frames >= 2:
                duration = t - self.cur["t0"]
                rep = {"index": len(self.targets[self.target]) + 1, "target": self.target,
                       "duration": duration, "ok": duration >= PRESS_TEMPLATE["duration_range"][0]}
                self.targets[self.target].append(rep)
                self._record(rep)
                self.cur = None
                self.press_frames = self.release_frames = 0
                self.phase = "释放"
                self.hint = "%s目标第%d次" % ("A" if self.target == "a" else "B", len(self.targets[self.target]))
        self.live = "目标A %d/4，目标B %d/4" % (len(self.targets["a"]), len(self.targets["b"]))

    def _drop(self, msg):
        self.cur = None
        self.phase = "idle"
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"target_zone": self.target, "target_a_count": len(self.targets["a"]),
                  "target_b_count": len(self.targets["b"])})
        return s


REGISTRY = [RubBackOfHand, FistOpenClose, FingerStretch, CupTransfer,
            FingerAbduction, FingerPress]
