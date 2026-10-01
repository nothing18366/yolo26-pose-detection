"""偏瘫康复操动作标准检测（Qt 界面）。

页0 患者信息 -> 页1 选择动作 -> 页2 实时检测（双模型）-> 页3 汇总。
动作判定逻辑在 actions/ 包，几何计算在 metrics.py。
"""

import sys
import time

import cv2
import numpy as np

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

from actions import REGISTRY, ActionContext, FingerStretch
from inference import PoseEngine, TEMPLATE_NAMES
from video_templates import load as load_template


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


class CaptureThread(QThread):
    frame = Signal(QImage)
    metrics = Signal(dict)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        self.action_key = None
        self.affected_side = None
        self.guidance_rois = {}
        self.engine = None

    def set_action_key(self, key, affected_side=None):
        self.action_key = key
        self.affected_side = affected_side
        template_name = TEMPLATE_NAMES.get(key)
        self.guidance_rois = load_template(template_name).get("guidance_rois", {}) if template_name else {}
        if self.engine is not None:
            self.engine.set_action_key(key, affected_side)

    def run(self):
        cap = None
        try:
            self.engine = PoseEngine()
            if self.action_key:
                self.engine.set_action_key(self.action_key, self.affected_side)
            cap = open_camera()
            if cap is None:
                self.error.emit("没找到可用的摄像头，请检查 系统设置 → 隐私与安全性 → 摄像头 里的授权")
                return

            misses = 0
            while self._running:
                ok, img = cap.read()
                if not ok:
                    misses += 1
                    if misses == 30:
                        self.error.emit("读不到画面，摄像头可能被别的程序占用（比如没关掉的上一个窗口）")
                    time.sleep(0.05)
                    continue
                misses = 0
                canvas, frame = self.engine.process(img, time.monotonic())
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(cv2.flip(canvas, 1), cv2.COLOR_BGR2RGB)
                self.frame.emit(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())
                self.metrics.emit(frame)
        except Exception as e:
            self.error.emit("摄像头或推理运行失败：%s" % e)
        finally:
            if cap is not None:
                cap.release()
            if self.engine is not None:
                self.engine.close()
                self.engine = None

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
        self.action = cls(ctx)
        if self.thread is not None:
            self.thread.set_action_key(self.action.key, ctx.affected)
        self.running = False
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
