"""Action state and result formatting for one training session."""

from rehab_service.recognition.actions import REGISTRY, ActionContext, FingerStretch

ACTIONS = {cls.key: cls for cls in REGISTRY}

def action_detail(action):
    if isinstance(action, FingerStretch):
        return "背伸 %d 次 / 掌屈 %d 次" % (
            len(action.modes["ext"]), len(action.modes["flex"]))
    if action.key == "cup":
        return "端起 %d / 放下 %d" % (action.lift_count, action.place_count)
    if action.key == "abduction":
        return "拇指 %d / 食指 %d" % (
            len(action.modes["thumb"]), len(action.modes["index"]))
    if action.key == "press":
        return "目标A %d / 目标B %d" % (
            len(action.targets["a"]), len(action.targets["b"]))
    return "-"


class TrainingSession:
    def __init__(self, payload):
        patient = payload.get("patient") or {}
        side = patient.get("affected_side")
        action_key = payload.get("action")
        if side not in ("left", "right"):
            raise ValueError("患侧必须是 left 或 right")
        if action_key not in ACTIONS:
            raise ValueError("不支持的动作：%s" % action_key)
        forearm_cm = float(patient.get("forearm_cm", 0))
        target_reps = int(patient.get("target_reps", 0))
        if not 10.0 <= forearm_cm <= 45.0:
            raise ValueError("前臂长度必须在 10-45 cm")
        if not 1 <= target_reps <= 100:
            raise ValueError("目标次数必须在 1-100")
        self.ctx = ActionContext(
            str(patient.get("name") or "未填写"), side, forearm_cm, target_reps)
        cls = ACTIONS[action_key]
        self.action = cls(self.ctx)

    def update(self, frame):
        self.ctx.update_forearm(frame["pose"])
        self.action.update(frame)
        status = self.action.status()
        status.update({
            "type": "result",
            "detail": action_detail(self.action),
            "forearm_px": self.ctx.forearm_px,
            "affected_side": self.ctx.affected,
            "healthy_side": self.ctx.healthy,
            "hand_tracking": frame.get("hand_tracking", {}),
            "tracking_confidence": frame.get("tracking_confidence", 0.0),
            "processing_ms": frame.get("processing_ms", 0.0),
        })
        return status

    def summary(self):
        result = self.action.summary()
        result.update({"type": "summary", "detail": action_detail(self.action)})
        return result
