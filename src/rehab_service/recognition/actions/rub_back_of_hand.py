from .common import Action, ELBOW, PAUSE_MAX_S, RUB_CONTACT_RATIO, RUB_FAR_RATIO, RUB_HEALTHY_MOTION_ADVANTAGE, RUB_KEEP_CONTACT_RATIO, RUB_MAX_AFFECTED_MOTION, RUB_MAX_FOREARM_SCALE_CHANGE, RUB_MAX_HAND_POSE_GAP, RUB_MAX_LATERAL_RATIO, RUB_MAX_PROJECTED_SPAN_RATIO, RUB_MAX_S, RUB_MIN_HEALTHY_MOTION, RUB_MIN_MOVE_RATIO, RUB_MIN_RATIO, RUB_MIN_S, RUB_MISSING_TOL_S, RUB_ORIGIN_TOLERANCE, RUB_RETURN_SPAN_TOLERANCE, RUB_START_HAND_CONTACT_RATIO, RUB_START_REGION, WRIST, dist, elbow_px, forearm_axis, get_hand, hand_state, line_angle, np, pose_pt, project_local, wrist_px

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
        aff_wrist = wrist_px(f, ctx.affected)
        aff_elbow = elbow_px(f, ctx.affected)
        if aff_wrist is None:
            aff_wrist = ctx.wrist_point
        if aff_elbow is None:
            aff_elbow = ctx.elbow_point
        healthy = get_hand(f, ctx.healthy)
        if aff_wrist is None or aff_elbow is None or healthy is None:
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

        axis, L = forearm_axis(aff_elbow, aff_wrist)
        if L < 20:
            self._drop("看不到患侧前臂轴线")
            return

        p = healthy["palm"]
        s, lateral = project_local(p, aff_wrist, axis, np.array([-axis[1], axis[0]]))
        s, perp = s / L, abs(lateral) / L
        in_axis = -1.25 <= s <= 1.9
        contact = perp <= RUB_CONTACT_RATIO and in_axis

        if self.cur is None:
            affected = get_hand(f, ctx.affected)
            near_hand = (affected is None or hand_state(affected) != "fresh" or
                         min(dist(p, point) for point in affected["pts"]) <=
                         RUB_START_HAND_CONTACT_RATIO * L)
            start_contact = (state == "fresh" and contact and -1.1 <= s <= RUB_START_REGION
                             and near_hand)
            if start_contact:
                healthy_ref_i = 5 if ctx.healthy == "left" else 6
                healthy_ref = pose_pt(f, healthy_ref_i, 0.6)
                if healthy_ref is None:
                    healthy_ref_i = ELBOW[ctx.healthy]
                    healthy_ref = pose_pt(f, healthy_ref_i, 0.6)
                affected_ref_i = 5 if ctx.affected == "left" else 6
                affected_ref = pose_pt(f, affected_ref_i, 0.6)
                if affected_ref is None:
                    affected_ref_i = ELBOW[ctx.affected]
                    affected_ref = pose_pt(f, affected_ref_i, 0.6)
                self.cur = {
                    "t0": t,
                    "t_last": t,
                    "p_prev": p,
                    "w0": aff_wrist.copy(),
                    "L": L,
                    "scale_off_frames": 0,
                    "s_start": s,
                    "s_peak": s,
                    "s_smooth": s,
                    "phase": "contact",
                    "far_frames": 0,
                    "return_frames": 0,
                    "max_angle": 0.0,
                    "max_lateral": perp,
                    "lateral_samples": [perp],
                    "max_drift": 0.0,
                    "max_pause": 0.0,
                    "lost_since": None,
                    "healthy_ref_i": healthy_ref_i if healthy_ref is not None else None,
                    "affected_ref_i": affected_ref_i if affected_ref is not None else None,
                    "healthy_rel0": p - healthy_ref if healthy_ref is not None else None,
                    "affected_rel0": aff_wrist - affected_ref if affected_ref is not None else None,
                    "healthy_motion": [],
                    "affected_motion": [],
                    "hand_pose_gaps": [],
                }
                self.phase = "contact"
                self.hint = "贴住了，朝肘部方向滑"
            else:
                self.hint = "把健侧手掌回到患侧腕部起点，再向肘部滑" if contact else "把健侧手掌贴到患侧手背上"
            return

        cur = self.cur
        scale = max(L / cur["L"], cur["L"] / L)
        cur["scale_off_frames"] = cur["scale_off_frames"] + 1 if scale > RUB_MAX_FOREARM_SCALE_CHANGE else 0
        if cur["scale_off_frames"] >= 4:
            self._drop("前臂角度变化过大，请稳定后重新开始")
            return
        cur["s_smooth"] = 0.55 * s + 0.45 * cur["s_smooth"]
        s = cur["s_smooth"]
        keep_contact = perp <= RUB_KEEP_CONTACT_RATIO and in_axis
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
        cur["max_drift"] = max(cur["max_drift"], dist(aff_wrist, cur["w0"]) / cur["L"])
        cur["max_lateral"] = max(cur["max_lateral"], perp)
        cur["lateral_samples"].append(perp)
        if state == "fresh" and cur["healthy_ref_i"] is not None:
            ref = pose_pt(f, cur["healthy_ref_i"], 0.6)
            if ref is not None:
                cur["healthy_motion"].append(dist(p - ref, cur["healthy_rel0"]) / cur["L"])
        if cur["affected_ref_i"] is not None:
            ref = pose_pt(f, cur["affected_ref_i"], 0.6)
            if ref is not None:
                cur["affected_motion"].append(dist(aff_wrist - ref, cur["affected_rel0"]) / cur["L"])
        affected_hand = get_hand(f, ctx.affected)
        pose_wrist = pose_pt(f, WRIST[ctx.affected], 0.6)
        if affected_hand is not None and hand_state(affected_hand) == "fresh" and pose_wrist is not None:
            cur["hand_pose_gaps"].append(dist(affected_hand["wrist"], pose_wrist) / L)
        if cur["phase"] == "contact":
            if s >= cur["s_start"] + RUB_MIN_MOVE_RATIO:
                cur["phase"] = "outbound"
                self.phase = "去程"
                self.hint = "去程中，继续向肘部方向滑"
        elif cur["phase"] == "outbound":
            cur["s_peak"] = max(cur["s_peak"], s)
            span = cur["s_peak"] - cur["s_start"]
            if span > RUB_MAX_PROJECTED_SPAN_RATIO:
                self._drop("关键点跳变，请重新贴住患侧手背")
                return
            self.live = "去程 %.0f%% 前臂长" % (span * 100)
            cur["far_frames"] = cur["far_frames"] + 1 if span >= RUB_FAR_RATIO else 0
            if cur["far_frames"] >= 2:
                cur["phase"] = "return"
                self.phase = "回程"
                self.hint = "达到远端，沿原路返回腕部"
            elif cur["s_peak"] - s >= 0.05:
                self.hint = "方向已改变，但还未到远端，请继续向肘部方向滑"
        else:
            span = cur["s_peak"] - cur["s_start"]
            self.live = "回程 %.0f%% 前臂长" % (span * 100)
            origin = cur["s_start"] + max(RUB_ORIGIN_TOLERANCE, RUB_RETURN_SPAN_TOLERANCE * span)
            cur["return_frames"] = cur["return_frames"] + 1 if s <= origin else 0
            if cur["return_frames"] >= 2:
                cur["s_return"] = s
                self._finish(t)

    def _finish(self, t):
        cur = self.cur
        span = cur["s_peak"] - cur["s_start"]
        healthy_motion = float(np.percentile(cur["healthy_motion"], 90)) if cur["healthy_motion"] else None
        affected_motion = float(np.percentile(cur["affected_motion"], 90)) if cur["affected_motion"] else None
        hand_pose_gap = float(np.median(cur["hand_pose_gaps"])) if cur["hand_pose_gaps"] else None
        if hand_pose_gap is not None and hand_pose_gap > RUB_MAX_HAND_POSE_GAP:
            self._drop("患侧手与手腕关键点不一致，请确认患侧选择")
            return
        if healthy_motion is not None and affected_motion is not None and (
            healthy_motion < RUB_MIN_HEALTHY_MOTION or
            (affected_motion > RUB_MAX_AFFECTED_MOTION and
             healthy_motion < RUB_HEALTHY_MOTION_ADVANTAGE * affected_motion)
        ):
            self._drop("健侧手没有完成相对滑动，请确认患侧选择")
            return
        rep = {
            "index": len(self.reps) + 1,
            "start_time": cur["t0"],
            "end_time": t,
            "distance_cm": span * self.ctx.forearm_cm,
            "ratio": span,
            "outbound_ratio": span,
            "return_ratio": max(0.0, cur["s_peak"] - cur["s_return"]),
            "angle": cur["max_angle"],
            # A single occlusion frame can put the palm far off-axis. Use a
            # robust within-cycle percentile for the template check, while
            # retaining the raw maximum for diagnostics.
            "lateral_ratio": float(np.percentile(cur["lateral_samples"], 90)),
            "raw_lateral_ratio": cur["max_lateral"],
            "drift_ratio": cur["max_drift"],
            "healthy_motion_ratio": healthy_motion,
            "affected_motion_ratio": affected_motion,
            "hand_pose_gap_ratio": hand_pose_gap,
            "duration": t - cur["t0"],
            "pause": cur["max_pause"],
        }
        rep["ok"] = (
            rep["ratio"] >= RUB_MIN_RATIO
            and rep["lateral_ratio"] <= RUB_MAX_LATERAL_RATIO
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
