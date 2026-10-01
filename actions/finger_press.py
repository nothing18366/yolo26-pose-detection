from .common import Action, FINGERS, PRESS_TEMPLATE, frame_size, get_hand, tracking_guard

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
