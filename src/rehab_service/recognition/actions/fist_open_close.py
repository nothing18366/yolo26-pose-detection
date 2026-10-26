from .common import Action, DRIFT_MAX_RATIO, FIST_MAX_S, FIST_MIN_S, FIST_TEMPLATE, FOUR, PAUSE_MAX_S, curl_metric, dist, fingers_closed, fingers_open, get_hand, hand_is_closed, hand_is_open, hand_state

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
