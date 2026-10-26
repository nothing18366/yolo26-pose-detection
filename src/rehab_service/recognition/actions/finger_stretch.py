from .common import Action, STRETCH_MODE_NAME, STRETCH_TEMPLATE, WRIST, get_hand, hand_state, np, pose_pt, stretch_features

class FingerStretch(Action):
    """Automatic, independent counts for observed extension/flexion push-return cycles."""

    key = "stretch"
    name = "动作3 牵伸手指"
    desc = "健侧辅助患侧指尖推压再回位，自动区分背伸和掌屈，分别累计"
    template = STRETCH_TEMPLATE

    def reset(self):
        super().reset()
        self.modes = {"ext": [], "flex": []}
        self.mode = None
        self.flip_since = None
        self.direction_since = None
        self.direction_candidate = None
        self._drop("双手进入画面，患侧四指向上准备背伸、向下准备掌屈")

    def update(self, f):
        t = f["t"]
        aff = get_hand(f, self.ctx.affected)
        heal = get_hand(f, self.ctx.healthy)
        assisting_wrist = pose_pt(f, WRIST[self.ctx.healthy], .7)
        wrist_only = (self.mode is not None and assisting_wrist is not None and
                      (heal is None or hand_state(heal) != "fresh"))
        if wrist_only:
            heal = {"side": self.ctx.healthy, "pts": assisting_wrist.reshape(1, 2), "state": "fresh"}
        states = [hand_state(h) for h in (aff, heal) if h is not None]
        if aff is None or heal is None or any(state != "fresh" for state in states):
            self.tracking_state = "手部丢失" if aff is None or heal is None or "lost" in states else "遮挡桥接"
            if self.missing_since is None:
                self.missing_since = t
            self.enter_since = self.return_since = None
            if t - self.missing_since > STRETCH_TEMPLATE["missing_frame_tolerance"]:
                self._drop("双手遮挡过久，请恢复关键点后重新推压")
            return
        self.missing_since = None
        self.tracking_state = "健侧腕部辅助" if wrist_only else "正常"
        feature = stretch_features(aff, heal, f.get("pose"))
        if feature is None:
            self._drop("患侧手部关键点不稳定，请调整角度")
            return
        self.samples = [(ts, v) for ts, v in self.samples
                        if t - ts <= STRETCH_TEMPLATE["smoothing_duration"]]
        self.samples.append((t, feature))
        motion = {key: float(np.median([v["motion"][key] for _, v in self.samples if key in v["motion"]]))
                  for key in feature["motion"]}
        if self.mode != "flex":
            motion.pop("wrist_gap", None)
        self.current_motion = motion
        self.current_length = feature["length"]
        gap = float(np.median([v["gap"] for _, v in self.samples]))
        direction = float(np.median([v["direction"] for _, v in self.samples]))
        self.current_direction = direction
        assisting_values = [abs(v["assisting_direction"]) for _, v in self.samples
                            if v["assisting_direction"] is not None]
        assisting = float(np.median(assisting_values)) if assisting_values else 1.0
        contact = gap <= STRETCH_TEMPLATE["contact_gap_ratio"]
        opposite = (not wrist_only and self.mode is not None and
                    assisting <= STRETCH_TEMPLATE["max_assisting_projection"] and
                    motion["curl"] >= STRETCH_TEMPLATE["direction_neutral_curl"] and
                    direction * (-1 if self.mode == "ext" else 1) < -STRETCH_TEMPLATE["direction_clear_projection"])
        if opposite:
            if self.flip_since is None:
                self.flip_since = t
            if t - self.flip_since >= STRETCH_TEMPLATE["direction_switch_duration"]:
                self._drop("已观察到相反准备方向，等待再次接触", reset_direction=True)
                return
        else:
            self.flip_since = None
        if not contact:
            if self.gap_since is None:
                self.gap_since = t
            if t - self.gap_since >= STRETCH_TEMPLATE["release_duration"]:
                if self.cur is not None and t - self.cur["t0"] >= STRETCH_TEMPLATE["min_push_duration"]:
                    self._finish(t)
                self._drop("健侧手接近患侧指尖后，再推压并回位")
            return
        self.gap_since = None

        if self.mode is None or self.reference is None:
            if self.mode is None:
                if (abs(direction) < STRETCH_TEMPLATE["direction_clear_projection"] or
                        assisting > STRETCH_TEMPLATE["max_assisting_projection"]):
                    self.direction_since = None
                    self.hint = "方向不明确，请让患侧四指朝向清晰可见"
                    return
                candidate = "ext" if direction < 0 else "flex"
                if self.direction_candidate != candidate or self.direction_since is None:
                    self.direction_candidate, self.direction_since = candidate, t
                if t - self.direction_since < STRETCH_TEMPLATE["direction_confirm_duration"]:
                    return
                self.mode = candidate
            self.reference = dict(motion)
            self.hand_length0 = feature["length"]
            self.phase = "接触"
            self.hint = "%s：缓慢推压，再回到准备位置" % STRETCH_MODE_NAME[self.mode]
            return

        scale_change = max(feature["length"] / self.hand_length0, self.hand_length0 / feature["length"])
        if scale_change > STRETCH_TEMPLATE["max_hand_scale_change"]:
            self._drop("翻掌或机位变化中，请稳定后重新开始")
            return
        thresholds = STRETCH_TEMPLATE["motion_thresholds"]
        if self.cur is None:
            for key, value in motion.items():
                self.reference.setdefault(key, value)
            for key, sign in self.polarities.items():
                if key in motion and (motion[key] - self.reference[key]) * sign < 0:
                    self.reference[key] = motion[key]
            candidates = {key: (motion[key] - value) / thresholds[key]
                          for key, value in self.reference.items() if key in motion}
            if not candidates:
                return
            driver = max(candidates, key=lambda key: candidates[key] * self.polarities[key]
                         if key in self.polarities else abs(candidates[key]))
            delta = candidates[driver]
            evidence = delta * self.polarities[driver] >= 1 if driver in self.polarities else abs(delta) >= 1
            if evidence:
                if self.enter_since is None:
                    self.enter_since = t
                if t - self.enter_since >= STRETCH_TEMPLATE["stable_duration"]:
                    sign = self.polarities.get(driver, 1 if delta > 0 else -1)
                    self.cur = {"t0": self.enter_since, "driver": driver, "sign": sign,
                                "base": self.reference[driver], "peak": abs(motion[driver] - self.reference[driver]),
                                "direction_scores": [], "wrist_only_frames": 0, "frames": 0}
                    self.phase = "推压/保持"
                    self.return_since = None
            else:
                self.enter_since = None
            self.live = "%s等待推压" % STRETCH_MODE_NAME[self.mode]
            return

        cur = self.cur
        cur["frames"] += 1
        cur["wrist_only_frames"] += int(wrist_only)
        cur["direction_scores"].append(feature["direction"] * (-1 if self.mode == "ext" else 1))
        if cur["driver"] not in motion:
            if t - cur["t0"] > STRETCH_TEMPLATE["max_candidate_duration"]:
                self._drop("姿态关键点不稳定，请重新开始一次")
            return
        directed = (motion[cur["driver"]] - cur["base"]) * cur["sign"]
        cur["peak"] = max(cur["peak"], directed)
        elapsed = t - cur["t0"]
        unit = "%前臂长" if cur["driver"] == "wrist_gap" else "°"
        self.live = "%s推压 %.1fs / 幅度 %.1f%s" % (STRETCH_MODE_NAME[self.mode], elapsed, cur["peak"], unit)
        if elapsed > STRETCH_TEMPLATE["max_candidate_duration"]:
            self._drop("推压未回位，请重新开始一次")
            return
        returned = directed <= cur["peak"] * STRETCH_TEMPLATE["return_fraction"]
        if returned:
            if self.return_since is None:
                self.return_since = t
            if t - self.return_since >= STRETCH_TEMPLATE["return_stable_duration"]:
                if elapsed >= STRETCH_TEMPLATE["min_push_duration"]:
                    self._finish(t)
                else:
                    self._drop("推压太短，本次不计数")
        else:
            self.return_since = None

    def _finish(self, t):
        cur = self.cur
        direction_ok = (bool(cur["direction_scores"]) and
                        float(np.percentile(cur["direction_scores"], 80)) >= STRETCH_TEMPLATE["cycle_direction_projection"])
        if (not direction_ok and self.current_motion["curl"] >= STRETCH_TEMPLATE["direction_neutral_curl"] and
                self.current_direction * (-1 if self.mode == "ext" else 1) < -STRETCH_TEMPLATE["transition_projection"]):
            self._drop("本次方向不明确，翻掌过渡不计数")
            return
        rep = {"index": len(self.modes[self.mode]) + 1, "mode": self.mode,
               "start_time": cur["t0"], "end_time": t, "duration": t - cur["t0"],
               "feature": cur["driver"], "amplitude": cur["peak"],
               "confirmed_return": self.gap_since is None,
               "wrist_assistance_ratio": cur["wrist_only_frames"] / max(cur["frames"], 1),
               "ok": direction_ok and self.gap_since is None and
                     cur["wrist_only_frames"] <= STRETCH_TEMPLATE["max_wrist_assistance_ratio"] * cur["frames"]}
        self.modes[self.mode].append(rep)
        self._record(rep)
        self.reference = dict(self.current_motion)
        self.reference[cur["driver"]] = cur["base"]
        self.polarities[cur["driver"]] = cur["sign"]
        self.hand_length0 = self.current_length
        self.cur = None
        self.enter_since = self.return_since = None
        self.phase = "回位"
        self.hint = "%s第 %d 次：%s" % (STRETCH_MODE_NAME[self.mode], rep["index"],
                                      "完成" if rep["ok"] else "动作证据需复核")

    def _drop(self, msg, reset_direction=False):
        self.cur = None
        if reset_direction:
            self.mode = None
            self.flip_since = None
            self.direction_since = None
            self.direction_candidate = None
        self.reference = None
        self.polarities = {}
        self.enter_since = self.return_since = None
        self.missing_since = self.gap_since = None
        self.samples = []
        self.phase = "等待准备"
        self.live = ""
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"mode": STRETCH_MODE_NAME.get(self.mode, "自动识别"),
                  "ext_count": len(self.modes["ext"]), "flex_count": len(self.modes["flex"])})
        return s
