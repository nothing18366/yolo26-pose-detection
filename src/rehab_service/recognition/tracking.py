"""Cross-frame hand assignment with bounded short-occlusion bridging."""

import numpy as np


class HandTracker:
    def __init__(self, max_missing_s=0.35):
        self.max_missing_s = float(max_missing_s)
        self.last = {}
        self.missing = {"left": 0.0, "right": 0.0}

    @staticmethod
    def _copy(hand, side, state="fresh", age=0.0, predicted=False):
        out = dict(hand)
        out["side"] = side
        out["state"] = state
        out["stale"] = state != "fresh"
        out["predicted"] = bool(predicted)
        out["age"] = float(age)
        return out

    @staticmethod
    def _shifted(hand, velocity, age, side, state):
        out = HandTracker._copy(hand, side, state=state, age=age, predicted=True)
        shift = np.asarray(velocity, dtype=float) * float(age)
        if "pts" in out:
            out["pts"] = np.asarray(out["pts"], dtype=float) + shift
        if "xyz" in out:
            xyz = np.asarray(out["xyz"], dtype=float).copy()
            xyz[:, :2] += shift
            out["xyz"] = xyz
        out["velocity"] = np.asarray(velocity, dtype=float).copy()
        return out

    @staticmethod
    def _distance(hand, previous):
        return float(np.linalg.norm(np.asarray(hand["pts"])[0] - np.asarray(previous["pts"])[0]))

    def _assign_single(self, hand, previous):
        explicit = str(hand.get("side", "")).lower()
        if explicit in ("left", "right"):
            return explicit
        if previous:
            side = min(previous, key=lambda s: self._distance(hand, previous[s]))
            return side
        cat = str(hand.get("cat", "")).lower()
        return "right" if cat == "left" else "left"

    def assign(self, hands, pose=None, frame_width=None):
        """Assign detected hands without changing cross-frame tracking state."""
        hands = list(hands)
        assigned = {}
        previous = {s: h for s, h in self.last.items()}
        pose_wrists = {}
        if pose is not None and len(pose) > 10:
            for side, index in (("left", 9), ("right", 10)):
                if pose[index, 2] >= 0.6:
                    pose_wrists[side] = np.asarray(pose[index, :2], dtype=float)
        separation = (np.linalg.norm(pose_wrists["left"] - pose_wrists["right"])
                      if len(pose_wrists) == 2 else 0.0)
        min_separation = max(35.0, 80.0 * frame_width / 720.0) if frame_width else 80.0
        pose_clear = separation >= min_separation

        if len(hands) == 2:
            explicit_sides = {h.get("side") for h in hands}
            if explicit_sides == {"left", "right"}:
                left_i = next(i for i, h in enumerate(hands) if h["side"] == "left")
                right_i = 1 - left_i
                pose_clear = True
            elif pose_clear:
                pairs = [(0, 1), (1, 0)]
                costs = [sum(np.linalg.norm(hands[i]["pts"][0] - pose_wrists[side])
                             for i, side in zip(pair, ("left", "right"))) for pair in pairs]
                left_i, right_i = pairs[int(np.argmin(costs))]
                if any(np.linalg.norm(hands[i]["pts"][0] - pose_wrists[side]) > separation * 0.55
                       for i, side in zip((left_i, right_i), ("left", "right"))):
                    pose_clear = False
            if not pose_clear and all(s in previous for s in ("left", "right")):
                pairs = [(0, 1), (1, 0)]
                costs = []
                for li, ri in pairs:
                    costs.append(
                        np.linalg.norm(hands[li]["pts"][0] - previous["left"]["pts"][0])
                        + np.linalg.norm(hands[ri]["pts"][0] - previous["right"]["pts"][0])
                    )
                left_i, right_i = pairs[int(np.argmin(costs))]
            elif not pose_clear:
                left_i, right_i = sorted(range(2), key=lambda i: hands[i]["pts"][0][0], reverse=True)
            assigned["left"] = hands[left_i]
            assigned["right"] = hands[right_i]
        elif len(hands) == 1:
            side = self._assign_single(hands[0], previous)
            if pose_clear and hands[0].get("side") not in ("left", "right"):
                near = min(pose_wrists, key=lambda s: np.linalg.norm(hands[0]["pts"][0] - pose_wrists[s]))
                if np.linalg.norm(hands[0]["pts"][0] - pose_wrists[near]) < separation * 0.35:
                    side = near
            assigned[side] = hands[0]

        return assigned

    def update(self, hands, t, pose=None, frame_width=None):
        assigned = self.assign(hands, pose, frame_width)
        previous = self.last.copy()

        result = []
        for side, hand in assigned.items():
            previous_hand = previous.get(side)
            now = float(t)
            old_t = float(previous_hand.get("_t", now)) if previous_hand else now
            dt = max(now - old_t, 1e-3)
            wrist = np.asarray(hand["pts"])[0]
            old_wrist = np.asarray(previous_hand["pts"])[0] if previous_hand else wrist
            velocity = (wrist - old_wrist) / dt if previous_hand else np.zeros(2)
            current = self._copy(hand, side, state="fresh", age=0.0, predicted=False)
            current["velocity"] = velocity.astype(float)
            current["_t"] = now
            self.last[side] = current
            self.missing[side] = 0.0
            result.append(current)
        for side in ("left", "right"):
            if side in assigned:
                continue
            if side not in self.last:
                continue
            age = t - self.last[side].get("_t", t)
            self.missing[side] = age
            state = "bridged" if age <= self.max_missing_s else "lost"
            velocity = self.last[side].get("velocity", np.zeros(2))
            result.append(self._shifted(self.last[side], velocity, age, side, state))
        return result
