from .common import ABDUCTION_TEMPLATE, Action, get_hand, hand_event_ok, hand_state, isolated_finger_features, np

class FingerAbduction(Action):
    key = "abduction"
    name = "动作5 手指外展"
    desc = "患侧拇指或食指单独伸出再收回，分别计数，可站立或悬空训练"
    template = ABDUCTION_TEMPLATE

    def reset(self):
        super().reset()
        self.mode = "both"
        self.modes = {"thumb": [], "index": []}
        self.cur = {"thumb": None, "index": None}
        self.ready = {"thumb": False, "index": False}
        self.enter_since = {"thumb": None, "index": None}
        self.return_since = {"thumb": None, "index": None}
        self.feature_samples = []
        self.hint = "患侧手先收拢，再单独伸出拇指或食指并收回"
        self.missing_since = None

    def update(self, f):
        hand = get_hand(f, self.ctx.affected)
        t = f["t"]
        if hand is None or hand_state(hand) != "fresh":
            self.tracking_state = "手部丢失" if hand is None or hand_state(hand) == "lost" else "遮挡桥接"
            if self.missing_since is None:
                self.missing_since = t
            self.enter_since = {"thumb": None, "index": None}
            self.return_since = {"thumb": None, "index": None}
            self.feature_samples = []
            if t - self.missing_since > ABDUCTION_TEMPLATE["missing_frame_tolerance"]:
                self._drop("患侧手不可见，请收回手指后重新开始")
            else:
                self.hint = "等待患侧真实关键点恢复，预测点不确认计数"
            return
        self.missing_since = None
        self.tracking_state = "正常"
        feature = isolated_finger_features(hand)
        if feature is None:
            self._drop("手部关键点不稳定，请调整角度")
            return
        self.feature_samples = [(ts, sample) for ts, sample in self.feature_samples
                                if t - ts <= ABDUCTION_TEMPLATE["smoothing_duration"]]
        self.feature_samples.append((t, dict(feature)))
        for key in ("thumb", "thumb_gap", "index", "index_angle", "other_reach"):
            feature[key] = float(np.median([sample[key] for _, sample in self.feature_samples]))
        stable = ABDUCTION_TEMPLATE["stable_duration"]
        isolated = feature["other_reach"] <= ABDUCTION_TEMPLATE["max_other_finger_reach"]
        for mode in ("thumb", "index"):
            value = feature[mode]
            limits = ABDUCTION_TEMPLATE["shape_thresholds"][mode]
            neutral = value <= limits["return"]
            extended = isolated and value >= limits["enter"]
            if mode == "index":
                extended = (extended and feature["index_angle"] >= limits["min_pip_angle"] and
                            feature["thumb"] <= limits["max_thumb_projection"] and
                            value - feature["other_reach"] >= limits["min_neighbor_difference"])
            else:
                extended = (extended and feature["thumb_gap"] >= limits["min_gap"] and
                            feature["index"] <= ABDUCTION_TEMPLATE["shape_thresholds"]["index"]["enter"])
            if not isolated:
                self.cur[mode] = None
                self.ready[mode] = False
                self.enter_since[mode] = self.return_since[mode] = None
                self.hint = "请单独伸出拇指或食指，避免整只手张开"
                continue
            if self.cur[mode] is None:
                if neutral:
                    if self.return_since[mode] is None:
                        self.return_since[mode] = t
                    if t - self.return_since[mode] >= stable:
                        self.ready[mode] = True
                else:
                    self.return_since[mode] = None
                if extended and self.ready[mode]:
                    if self.enter_since[mode] is None or t - self.enter_since[mode] > ABDUCTION_TEMPLATE["missing_frame_tolerance"]:
                        self.enter_since[mode] = t
                    if t - self.enter_since[mode] >= stable:
                        self.cur[mode] = {"t0": self.enter_since[mode], "peak": value}
                        self.phase = "%s伸展" % ("拇指" if mode == "thumb" else "食指")
                        self.return_since[mode] = None
                elif neutral or not self.ready[mode]:
                    self.enter_since[mode] = None
            else:
                self.cur[mode]["peak"] = max(self.cur[mode]["peak"], value)
                if t - self.cur[mode]["t0"] > ABDUCTION_TEMPLATE.get("max_candidate_duration", 4.0):
                    self.cur[mode] = None
                    self.ready[mode] = False
                    self.enter_since[mode] = self.return_since[mode] = None
                    continue
                if neutral:
                    if self.return_since[mode] is None:
                        self.return_since[mode] = t
                else:
                    self.return_since[mode] = None
                if (self.return_since[mode] is not None and
                        t - self.return_since[mode] >= ABDUCTION_TEMPLATE["return_stable_duration"]):
                    duration = t - self.cur[mode]["t0"]
                    rep = {"index": len(self.modes[mode]) + 1, "mode": mode, "duration": duration,
                           "start_time": self.cur[mode]["t0"], "end_time": t,
                           "peak": self.cur[mode]["peak"], "feature_source": feature["source"],
                           "ok": hand_event_ok(duration, ABDUCTION_TEMPLATE["duration_range"])}
                    self.modes[mode].append(rep)
                    self._record(rep)
                    self.cur[mode] = None
                    self.enter_since[mode] = None
                    self.ready[mode] = True
                    self.phase = "回位"
                    self.hint = "本次%s完成" % ("拇指" if mode == "thumb" else "食指")
        self.live = "拇指 %d，食指 %d" % (len(self.modes["thumb"]), len(self.modes["index"]))

    def _drop(self, msg):
        self.cur = {"thumb": None, "index": None}
        self.ready = {"thumb": False, "index": False}
        self.enter_since = {"thumb": None, "index": None}
        self.return_since = {"thumb": None, "index": None}
        self.feature_samples = []
        self.phase = "idle"
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"mode": self.mode, "thumb_count": len(self.modes["thumb"]), "index_count": len(self.modes["index"])})
        return s
