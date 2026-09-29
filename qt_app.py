"""偏瘫康复操动作标准检测（Qt 界面）。

页0 患者信息 -> 页1 选择动作 -> 页2 实时检测（双模型）-> 页3 汇总。
动作判定逻辑在 actions.py（动作即插件），几何计算在 metrics.py。
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from actions import REGISTRY, STRETCH_MODE_NAME, ActionContext, FingerStretch
from metrics import finger_angles, palm_center, local_frame, ExpSmoother
from video_templates import load as load_template
from tracking import HandTracker

torch.set_num_threads(4)

MODEL = str(Path(__file__).with_name("yolo26n-pose.pt"))
HAND_MODEL = str(Path(__file__).with_name("hand_landmarker.task"))
HAND_LINKS = vision.HandLandmarksConnections.HAND_CONNECTIONS
HAND_LINE, HAND_DOT = (0, 255, 0), (0, 0, 255)
IMGSZ = 640  # 要更精确改 640（慢一倍）
POSE_EVERY_N_FRAMES = 2  # 姿态骨架每两帧更新一次，动作状态仍逐帧更新
SWAP_HANDEDNESS = True  # 摄像头给的是非镜像原图，MediaPipe 按镜像图假设，需要交换
TEMPLATE_NAMES = {
    "rub": "rub_hand_back",
    "fist": "fist_extension",
    "stretch": "finger_stretch",
    "cup": "cup_transfer",
    "abduction": "finger_abduction",
    "press": "finger_press",
}


def detect_hands_by_half(landmarker, img, previous_hands=None, guidance_rois=None):
    """Detect both hands, then search only the missing side's last ROI when needed."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def convert(det, x0=0, y0=0, cw=w, ch=h):
        out = []
        for i, hl in enumerate(det.hand_landmarks):
            pts = np.array([[lm.x * cw + x0, lm.y * ch + y0] for lm in hl], dtype=float)
            xyz = np.array([[lm.x * cw + x0, lm.y * ch + y0, lm.z * cw] for lm in hl], dtype=float)
            out.append({"cat": det.handedness[i][0].category_name, "pts": pts, "xyz": xyz})
        return out

    def detect(crop, x0=0, y0=0):
        ch, cw = crop.shape[:2]
        det = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.ascontiguousarray(crop)))
        return convert(det, x0, y0, cw, ch)

    result = convert(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))))
    if len(result) >= 2:
        return result

    regions = []

    def add_region(region):
        xa, ya, xb, yb = region
        if xb - xa <= 32 or yb - ya <= 32:
            return
        candidate = (max(0, xa), max(0, ya), min(w, xb), min(h, yb))
        # Overlapping fallback crops repeat the same expensive hand inference.
        if not any(
            max(0, min(candidate[2], old[2]) - max(candidate[0], old[0]))
            * max(0, min(candidate[3], old[3]) - max(candidate[1], old[1]))
            >= 0.7 * min(
                (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
                (old[2] - old[0]) * (old[3] - old[1]),
            )
            for old in regions
        ):
            regions.append(candidate)

    # The previous hand box is the most likely location of the missing hand;
    # try it before generic action ROIs to avoid unnecessary crop inference.
    for old in (previous_hands or {}).values():
        pts = np.asarray(old.get("pts"), dtype=float)
        if pts.size == 0:
            continue
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)
        pad_x, pad_y = max((x1 - x0) * 0.20, 32), max((y1 - y0) * 0.20, 32)
        xa, ya = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
        xb, yb = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y))
        add_region((xa, ya, xb, yb))
    for roi in (guidance_rois or {}).values():
        if len(roi) != 4:
            continue
        xa, ya, xb, yb = [int(v) for v in (roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h)]
        add_region((xa, ya, xb, yb))
    if not regions:
        regions = [(0, 0, w // 2, h), (w // 2, 0, w, h)]

    for x0, y0, x1, y1 in regions:
        for hand in detect(rgb[y0:y1, x0:x1], x0, y0):
            if all(np.linalg.norm(hand["pts"][0] - old["pts"][0]) > 35 for old in result):
                result.append(hand)
        if len(result) >= 2:
            break
    return result


def open_camera():
    """挑真正在出画面的设备。虚拟摄像头（如 OBS）没推流时会输出静态占位图，帧完全不变。"""
    for index in range(4):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        first = last = None
        for _ in range(6):
            ok, frame = cap.read()
            if ok:
                first = first if first is not None else frame
                last = frame
        if first is not None and np.abs(first.astype(np.int16) - last.astype(np.int16)).mean() > 0.01:
            print("使用摄像头 %d：%s %dx%d" % (index, cap.getBackendName(), last.shape[1], last.shape[0]))
            return cap
        cap.release()
    return None


def assign_sides(hands):
    """判定每只手属于患者左手还是右手。

    摄像头给的是非镜像画面（人面对镜头时，图像左侧 = 患者右手），
    所以两只手同时出现时按 x 排序最稳；只有一只手时退回 handedness（需交换）。
    """
    if len(hands) == 2:
        left_img, right_img = sorted(hands, key=lambda h: h["pts"][0][0])
        left_img["side"], right_img["side"] = "right", "left"
    elif len(hands) == 1:
        cat = hands[0]["cat"].lower()
        if SWAP_HANDEDNESS:
            cat = "right" if cat == "left" else "left"
        hands[0]["side"] = cat if cat in ("left", "right") else "right"


class CaptureThread(QThread):
    frame = Signal(QImage)
    metrics = Signal(dict)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        self.action_key = None
        self.guidance_rois = {}

    def set_action_key(self, key):
        self.action_key = key
        template_name = TEMPLATE_NAMES.get(key)
        self.guidance_rois = load_template(template_name).get("guidance_rois", {}) if template_name else {}

    def run(self):
        try:
            model = YOLO(MODEL)
            hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=HAND_MODEL, delegate=BaseOptions.Delegate.CPU),
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            )
        except Exception as e:
            self.error.emit("模型加载失败：%s" % e)
            return
        cap = open_camera()
        if cap is None:
            self.error.emit("没找到可用的摄像头，请检查 系统设置 → 隐私与安全性 → 摄像头 里的授权")
            return

        misses = 0
        smooth = {"left": ExpSmoother(), "right": ExpSmoother()}
        tracker = HandTracker(max_missing_s=0.35)
        frame_no = 0
        pose = None
        pose_canvas = None
        while self._running:
            ok, img = cap.read()
            if not ok:
                misses += 1
                if misses == 30:
                    self.error.emit("读不到画面，摄像头可能被别的程序占用（比如没关掉的上一个窗口）")
                time.sleep(0.05)
                continue
            misses = 0
            t = time.monotonic()
            frame_no += 1

            # 两个模型都跑在未翻转的原图上，保证左右手与 YOLO 左右一致；只在显示时翻转
            if pose_canvas is None or frame_no % POSE_EVERY_N_FRAMES == 1:
                result = model.predict(img, imgsz=IMGSZ, conf=0.5, verbose=False)[0]
                pose_canvas = result.plot()
                pose = result.keypoints.data[0].cpu().numpy() if len(result.keypoints.data) else None
            h, w = img.shape[:2]

            guidance = self.guidance_rois
            hlist = detect_hands_by_half(hands, img, tracker.last, guidance)
            hlist = tracker.update(hlist, t)
            for hd in hlist:
                if hd.get("state", "fresh") == "fresh":
                    hd["pts"] = smooth[hd["side"]].update(hd["pts"])
                hd["angles"] = finger_angles(hd["pts"])
                hd["palm"] = palm_center(hd["pts"])
                hd["wrist"] = hd["pts"][0].copy()
            canvas = pose_canvas.copy()
            for hd in hlist:
                if hd.get("state", "fresh") == "fresh":
                    for link in HAND_LINKS:
                        cv2.line(canvas, tuple(hd["pts"][link.start].astype(int)),
                                 tuple(hd["pts"][link.end].astype(int)), HAND_LINE, 2)
                    for p in hd["pts"].astype(int):
                        cv2.circle(canvas, tuple(p), 3, HAND_DOT, -1)
            rgb = cv2.cvtColor(cv2.flip(canvas, 1), cv2.COLOR_BGR2RGB)
            self.frame.emit(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())
            axis = None
            if pose is not None and len(pose) > 10:
                for side, ei, wi in (("left", 7, 9), ("right", 8, 10)):
                    if pose[ei, 2] >= .3 and pose[wi, 2] >= .3:
                        axis, _ = local_frame(pose[ei, :2], pose[wi, :2])[:2]
                        break
            hand_tracking = {
                h["side"]: {"state": h.get("state", "fresh"), "age": h.get("age", 0.0),
                            "predicted": bool(h.get("predicted"))}
                for h in hlist
            }
            self.metrics.emit({"t": t, "pose": pose, "hands": hlist,
                               "frame_size": (w, h),
                               "forearm_axis": axis,
                               "hand_tracking": hand_tracking,
                               "tracking_confidence": sum(h.get("state", "fresh") == "fresh" for h in hlist) / 2.0})
        hands.close()
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("偏瘫康复操动作标准检测")
        self.resize(1200, 760)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.patient = None
        self.ctx = None
        self.action = None
        self.running = False
        self.thread = None

        self.build_patient_page()
        self.build_action_page()
        self.build_detect_page()
        self.build_summary_page()
        self.stack.setCurrentIndex(0)

    # ---------------- 页0 患者信息 ----------------
    def build_patient_page(self):
        page = QWidget()
        form = QFormLayout(page)
        title = QLabel("① 患者信息")
        title.setStyleSheet("font-size:20px;font-weight:bold")
        form.addRow(title)

        self.in_name = QLineEdit("患者A")
        self.in_side = QComboBox()
        self.in_side.addItems(["左侧", "右侧"])
        self.in_forearm = QDoubleSpinBox()
        self.in_forearm.setRange(10.0, 45.0)
        self.in_forearm.setValue(25.0)
        self.in_forearm.setSuffix(" cm")
        self.in_target = QSpinBox()
        self.in_target.setRange(1, 100)
        self.in_target.setValue(10)
        self.in_target.setSuffix(" 次")

        form.addRow("姓名", self.in_name)
        form.addRow("患侧", self.in_side)
        form.addRow("前臂长度", self.in_forearm)
        form.addRow("目标次数", self.in_target)

        btn = QPushButton("下一步：选择动作")
        btn.clicked.connect(self.goto_actions)
        form.addRow(btn)
        self.stack.addWidget(page)

    def goto_actions(self):
        self.patient = {
            "name": self.in_name.text().strip() or "未填写",
            "side": "left" if self.in_side.currentIndex() == 0 else "right",
            "forearm_cm": self.in_forearm.value(),
            "target_reps": self.in_target.value(),
        }
        self.stack.setCurrentIndex(1)

    # ---------------- 页1 选择动作 ----------------
    def build_action_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        title = QLabel("② 选择康复动作")
        title.setStyleSheet("font-size:20px;font-weight:bold")
        v.addWidget(title)
        for cls in REGISTRY:
            btn = QPushButton("%s\n%s" % (cls.name, cls.desc))
            btn.setMinimumHeight(70)
            btn.clicked.connect(lambda _=False, c=cls: self.start_action(c))
            v.addWidget(btn)
        back = QPushButton("返回患者信息")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        v.addWidget(back)
        self.stack.addWidget(page)

    def start_action(self, cls):
        ctx = ActionContext(self.patient["name"], self.patient["side"],
                            self.patient["forearm_cm"], self.patient["target_reps"])
        self.ctx = ctx
        self.action = cls(ctx) if cls is not FingerStretch else cls(ctx, "auto")
        if self.thread is not None:
            self.thread.set_action_key(self.action.key)
        self.running = False
        self.cmb_mode.setVisible(getattr(self.action, "has_mode", False))
        if isinstance(self.action, FingerStretch):
            self.cmb_mode.blockSignals(True)
            self.cmb_mode.setCurrentIndex(0)
            self.cmb_mode.blockSignals(False)
        self.lb_action.setText(self.action.name)
        self.lb_target.setText(str(ctx.target_reps))
        self.lb_count.setText("0")
        self.lb_pass.setText("0")
        self.lb_hint.setText("确认患侧手和健侧手标签后，点“开始检测”")
        self.lb_position.setText("参考框仅用于引导，不限制检测")
        self.lb_live.setText("")
        self.stack.setCurrentIndex(2)

    # ---------------- 页2 实时检测 ----------------
    def build_detect_page(self):
        page = QWidget()
        h = QHBoxLayout(page)

        left = QVBoxLayout()
        self.video = QLabel("正在打开摄像头…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(820, 520)
        self.video.setStyleSheet("background:#111;color:#eee;font-size:16px")
        left.addWidget(self.video)
        self.lb_live = QLabel("")
        self.lb_live.setStyleSheet("color:#2a7;font-size:15px")
        left.addWidget(self.lb_live)
        h.addLayout(left, 3)

        right = QVBoxLayout()
        box = QGroupBox("本次检测")
        f = QFormLayout(box)
        self.lb_action = QLabel("-")
        self.lb_target = QLabel("-")
        self.lb_count = QLabel("0")
        self.lb_pass = QLabel("0")
        self.lb_phase = QLabel("待机")
        self.lb_tracking = QLabel("--")
        self.lb_position = QLabel("--")
        self.lb_sides = QLabel("患侧：- / 健侧：-")
        self.lb_forearm = QLabel("-")
        f.addRow("动作", self.lb_action)
        f.addRow("目标次数", self.lb_target)
        f.addRow("完成次数", self.lb_count)
        f.addRow("合格次数", self.lb_pass)
        f.addRow("当前阶段", self.lb_phase)
        f.addRow("手部标记", self.lb_sides)
        f.addRow("跟踪状态", self.lb_tracking)

        right.addWidget(box)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["自动识别", STRETCH_MODE_NAME["ext"], STRETCH_MODE_NAME["flex"]])
        self.cmb_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.cmb_mode.setVisible(False)
        right.addWidget(QLabel("动作3 方向"))
        right.addWidget(self.cmb_mode)

        self.lb_hint = QLabel("")
        self.lb_hint.setWordWrap(True)
        self.lb_hint.setStyleSheet("color:#333;font-size:15px;background:#f2f2f2;padding:8px")
        right.addWidget(self.lb_hint)
        right.addWidget(QLabel("安全提示：出现疼痛、明显痉挛或手腕抬起时，请停止并复核。"))

        self.btn_start = QPushButton("开始检测")
        self.btn_start.clicked.connect(self.start_running)
        self.btn_stop = QPushButton("结束并查看汇总")
        self.btn_stop.clicked.connect(self.finish_running)
        back = QPushButton("返回动作选择")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        for b in (self.btn_start, self.btn_stop, back):
            b.setMinimumHeight(38)
            right.addWidget(b)
        right.addStretch(1)
        h.addLayout(right, 2)
        self.stack.addWidget(page)

    def on_mode_changed(self, idx):
        if isinstance(self.action, FingerStretch):
            self.action.set_mode(("auto", "ext", "flex")[idx])

    def start_running(self):
        if not self.action:
            return
        self.action.reset()
        self.running = True
        self.lb_hint.setText("开始检测")
        self.lb_live.setText("")

    def finish_running(self):
        self.running = False
        self.show_summary()

    # ---------------- 页3 汇总 ----------------
    def build_summary_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        title = QLabel("③ 本次训练汇总")
        title.setStyleSheet("font-size:20px;font-weight:bold")
        v.addWidget(title)
        self.sum_labels = {}
        for key, text in [("name", "动作"), ("target", "目标次数"), ("count", "完成次数"),
                          ("passed", "合格次数"), ("rate", "合格率"), ("review", "需复核"), ("detail", "分项"),
                          ("total", "总耗时"), ("avg", "平均单次")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(text))
            lb = QLabel("-")
            lb.setStyleSheet("font-size:16px;font-weight:bold")
            row.addWidget(lb)
            row.addStretch(1)
            v.addLayout(row)
            self.sum_labels[key] = lb
        v.addWidget(QLabel("说明：前置斜拍存在透视压缩，距离/角度按 2D 图像平面近似。"))
        again = QPushButton("再做一次")
        again.clicked.connect(lambda: self.start_action(type(self.action)))
        choose = QPushButton("换动作")
        choose.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        info = QPushButton("返回患者信息")
        info.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        for b in (again, choose, info):
            b.setMinimumHeight(38)
            v.addWidget(b)
        v.addStretch(1)
        self.stack.addWidget(page)

    def show_summary(self):
        if not self.action:
            return
        s = self.action.summary()
        self.sum_labels["name"].setText(s["name"])
        self.sum_labels["target"].setText(str(s["target"]))
        self.sum_labels["count"].setText(str(s["count"]))
        self.sum_labels["passed"].setText(str(s["passed"]))
        self.sum_labels["rate"].setText("%.0f%%" % s["rate"])
        self.sum_labels["review"].setText(str(s.get("review", 0)))
        self.sum_labels["total"].setText("%.1f s" % s["total"])
        self.sum_labels["avg"].setText("%.2f s" % s["avg"])
        if isinstance(self.action, FingerStretch):
            self.sum_labels["detail"].setText("背伸 %d 次 / 掌屈 %d 次" % (
                len(self.action.modes["ext"]), len(self.action.modes["flex"])))
        elif getattr(self.action, "key", "") == "cup":
            self.sum_labels["detail"].setText("端起 %d / 放下 %d" % (self.action.lift_count, self.action.place_count))
        elif getattr(self.action, "key", "") == "abduction":
            self.sum_labels["detail"].setText("拇指 %d / 食指 %d" % (
                len(self.action.modes["thumb"]), len(self.action.modes["index"])))
        elif getattr(self.action, "key", "") == "press":
            self.sum_labels["detail"].setText("目标A %d / 目标B %d" % (
                len(self.action.targets["a"]), len(self.action.targets["b"])))
        else:
            self.sum_labels["detail"].setText("-")
        self.stack.setCurrentIndex(3)

    # ---------------- 信号处理 ----------------
    def start_capture(self):
        self.thread = CaptureThread()
        self.thread.frame.connect(self.on_frame)
        self.thread.metrics.connect(self.on_metrics)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_frame(self, qimg):
        pm = QPixmap.fromImage(qimg)
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_error(self, msg):
        self.lb_hint.setText(msg)
        self.video.setText(msg)

    def on_metrics(self, f):
        if self.ctx:
            self.ctx.update_forearm(f["pose"])
        if self.ctx and self.ctx.forearm_px:
            self.lb_forearm.setText("%.0f px" % self.ctx.forearm_px)
        if self.running and self.action:
            self.action.update(f)
            st = self.action.status()
            self.lb_count.setText(str(st["count"]))
            self.lb_pass.setText(str(st["passed"]))
            self.lb_hint.setText(st["hint"])
            self.lb_phase.setText(st.get("phase", "待机"))
            self.lb_tracking.setText("手部：%s" % st.get("tracking", "正常"))
            self.lb_position.setText(st.get("position_hint") or st.get("baseline_status") or "参考框仅用于引导")
            extra = st.get("mode", "")
            if isinstance(self.action, FingerStretch):
                extra = "%s  背伸%d / 掌屈%d" % (extra, st.get("ext_count", 0), st.get("flex_count", 0))
            elif getattr(self.action, "key", "") == "cup":
                extra = "端起%d / 放下%d" % (st.get("lift_count", 0), st.get("place_count", 0))
            elif getattr(self.action, "key", "") == "abduction":
                extra = "拇指%d / 食指%d" % (st.get("thumb_count", 0), st.get("index_count", 0))
            elif getattr(self.action, "key", "") == "press":
                extra = "目标A%d / 目标B%d" % (st.get("target_a_count", 0), st.get("target_b_count", 0))
            self.lb_live.setText(("%s  " % extra if extra else "") + st["live"])
            if st["count"] >= st["target"] and st["count"]:
                self.lb_hint.setText("已达到目标次数，可点“结束并查看汇总”")
        if self.ctx and not self.running:
            self.lb_phase.setText("待机")
        if f.get("tracking_confidence") is not None:
            elbow = "肘部沿用上一帧" if self.ctx and self.ctx.elbow_stale else "肘部正常"
            states = f.get("hand_tracking", {})
            side_state = " / ".join("%s:%s" % (s, v.get("state", "fresh")) for s, v in sorted(states.items()))
            self.lb_tracking.setText("手部 %.0f%% %s / %s" % (
                f["tracking_confidence"] * 100, side_state, elbow))
        if self.ctx:
            self.lb_sides.setText("患侧手：%s / 健侧手：%s" % (self.ctx.affected, self.ctx.healthy))

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.start_capture()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
