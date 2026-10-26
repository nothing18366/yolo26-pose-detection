from .common import Action, CUP_TEMPLATE, ELBOW, WRIST, dist, get_hand, hand_state, np, pose_pt

class CupTransfer(Action):
    key = "cup"
    name = "动作4 端放水杯"
    desc = "患侧手相对身体抬起和放下分别计数，可持杯或空手训练"
    template = CUP_TEMPLATE

    def reset(self):
        super().reset()
        self.lift_count = self.place_count = 0
        self.state = "prepare"
        self.cur = None
        self.reference_i = None
        self.scale = None
        self.body_scale = None
        self.baseline = None
        self.baseline_lateral = None
        self.start_t = None
        self.samples = []
        self.scale_samples = []
        self.enter_since = None
        self.missing_since = None
        self.last_event_t = -999.0
        self.baseline_status = "等待姿态对齐"
        self.position_hint = "让患侧肩、肘、腕尽量入镜，可持杯或空手"
        self.hint = "先保持准备位置，再抬起患侧手"

    def update(self, f):
        t = f["t"]
        side = self.ctx.affected
        if self.reference_i is None:
            shoulder_i = 5 if side == "left" else 6
            if pose_pt(f, shoulder_i, .4) is not None:
                self.reference_i = shoulder_i
            elif pose_pt(f, ELBOW[side], .4) is not None:
                self.reference_i = ELBOW[side]
        origin = pose_pt(f, self.reference_i, .4) if self.reference_i is not None else None
        wrist = pose_pt(f, WRIST[side], .4)
        if wrist is None:
            hand = get_hand(f, side)
            wrist = hand["wrist"] if hand is not None and hand_state(hand) == "fresh" else None
            self.tracking_state = "手部腕点辅助"
        else:
            self.tracking_state = "腕部姿态跟踪"
        elbow = pose_pt(f, ELBOW[side], .4)
        if origin is None or wrist is None or (self.scale is None and elbow is None):
            self.enter_since = None
            if self.missing_since is None:
                self.missing_since = t
            if t - self.missing_since > CUP_TEMPLATE["missing_frame_tolerance"]:
                self._drop("患侧姿态点丢失，请恢复肩肘腕后重新开始")
            return
        if self.missing_since is not None and t - self.missing_since > CUP_TEMPLATE["missing_frame_tolerance"]:
            self._drop("患侧姿态已恢复，重新对齐")
        self.missing_since = None

        vertical = np.array([0.0, 1.0])
        body_length = None
        pose = f.get("pose")
        if pose is not None and all(pose[i, 2] >= .4 for i in (5, 6, 11, 12)):
            body = (pose[11, :2] + pose[12, :2]) - (pose[5, :2] + pose[6, :2])
            if np.linalg.norm(body) > 20:
                vertical = body / np.linalg.norm(body)
                body_length = float(np.linalg.norm(body) / 2.0)
        shoulder = pose_pt(f, 5 if side == "left" else 6, .4)
        if elbow is not None and shoulder is not None:
            arm_length = max(dist(elbow, wrist), dist(shoulder, elbow))
            if np.dot(wrist - elbow, vertical) > CUP_TEMPLATE["max_lower_forearm_ratio"] * arm_length:
                self._drop("请先弯肘进入准备位置，垂臂姿态不计数")
                return
        position = float(np.dot(wrist - origin, vertical))
        lateral = float(np.dot(wrist - origin, [vertical[1], -vertical[0]]))
        self.samples = [sample for sample in self.samples if t - sample[0] <= CUP_TEMPLATE["smoothing_duration"]]
        self.samples.append((t, position, lateral))
        position, lateral = np.median([sample[1:] for sample in self.samples], axis=0)
        if self.scale is None:
            length = dist(elbow, wrist)
            if shoulder is not None:
                length = max(length, dist(shoulder, elbow))
            if length < 20:
                self.hint = "前臂投影太短，请让肘腕清晰可见"
                return
            if self.start_t is None:
                self.start_t = t
            self.scale_samples = [(ts, v) for ts, v in self.scale_samples
                                  if t - ts <= CUP_TEMPLATE["baseline_sample_duration"]]
            self.scale_samples.append((t, length))
            self.phase = "准备对齐"
            if t - self.start_t < CUP_TEMPLATE["baseline_sample_duration"]:
                return
            self.scale = float(np.median([v for _, v in self.scale_samples]))
            self.body_scale = body_length
            self.baseline = position / self.scale
            self.baseline_lateral = lateral / self.scale
            self.state = "lower"
            self.baseline_status = "身体相对坐标已建立"
        # ponytail: torso scale compensates zoom; without visible hips keep the initial arm scale.
        scale = self.scale * body_length / self.body_scale if body_length and self.body_scale else self.scale
        height = position / scale
        lateral /= scale
        stable = CUP_TEMPLATE["stable_duration"]
        if self.state == "lower":
            if height >= self.baseline:
                self.baseline, self.baseline_lateral = height, lateral
            up = self.baseline - height >= CUP_TEMPLATE["wrist_up_threshold"]
            if up and abs(lateral - self.baseline_lateral) > CUP_TEMPLATE["max_lateral_to_vertical"] * (self.baseline - height):
                self.baseline, self.baseline_lateral = height, lateral
                self.enter_since = None
                self.hint = "横向移位或取物不计端起，请上下移动患侧腕"
                return
            if up:
                if self.enter_since is None:
                    self.enter_since = t
                if t - self.enter_since >= stable and t - self.last_event_t >= CUP_TEMPLATE["min_event_interval"]:
                    self.state = "raised"
                    self.cur = {"t0": self.enter_since, "low": self.baseline, "high": height}
                    self.lift_count += 1
                    self._record({"index": self.lift_count, "event": "lift", "time": t,
                                  "duration": t - self.enter_since, "motion_ratio": self.baseline - height,
                                  "ok": True})
                    self.phase = "端起"
                    self.enter_since = None
                    self.last_event_t = t
            else:
                self.enter_since = None
        else:
            self.cur["high"] = min(self.cur["high"], height)
            return_distance = max(CUP_TEMPLATE["wrist_down_threshold"],
                                  min(CUP_TEMPLATE["max_return_ratio"],
                                      CUP_TEMPLATE["return_fraction"] * (self.cur["low"] - self.cur["high"])))
            down = height - self.cur["high"] >= return_distance
            if down:
                if self.enter_since is None:
                    self.enter_since = t
                if t - self.enter_since >= stable and t - self.last_event_t >= CUP_TEMPLATE["min_event_interval"]:
                    self.place_count += 1
                    self._record({"index": self.place_count, "event": "place", "time": t,
                                  "duration": t - self.cur["t0"], "motion_ratio": height - self.cur["high"],
                                  "ok": True})
                    self.state = "lower"
                    self.baseline = height
                    self.baseline_lateral = lateral
                    self.phase = "放下"
                    self.cur = None
                    self.enter_since = None
                    self.last_event_t = t
            else:
                self.enter_since = None
        self.hint = "继续抬起或放下患侧手，无需对准桌面区域"
        self.live = "端起 %d / 放下 %d" % (self.lift_count, self.place_count)

    def _drop(self, msg):
        self.state = "prepare"
        self.cur = None
        self.reference_i = self.scale = self.baseline = None
        self.baseline_lateral = None
        self.body_scale = None
        self.start_t = self.enter_since = self.missing_since = None
        self.samples = []
        self.scale_samples = []
        self.phase = "等待姿态"
        self.baseline_status = "等待重新对齐"
        self.hint = msg

    def status(self):
        s = super().status()
        s.update({"lift_count": self.lift_count, "place_count": self.place_count})
        return s
