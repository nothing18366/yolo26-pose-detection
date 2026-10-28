"""Qt local skeleton rendering and latest-frame capture without a physical camera."""

import os
from pathlib import Path
import sys
from threading import Event
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from rehab_service.desktop.app import CaptureThread, MainWindow
from rehab_service.desktop.skeleton_view import SkeletonVideoView

APP = QApplication.instance() or QApplication([])


def image(width, height):
    result = QImage(width, height, QImage.Format_RGB888)
    result.fill(QColor("#444444"))
    return result


def metrics(width=640, height=480):
    pose = np.full((17, 3), np.nan)
    pose[9] = [width * .25, height * .25, .9]
    return {"t": time.monotonic(), "pose": pose, "hands": [], "frame_size": (width, height)}


def test_fit_mirror_resize_portrait_and_expiry():
    view = SkeletonVideoView()
    view.resize(800, 800)
    view.set_frame(image(640, 480))
    view.set_pose(metrics())
    pixels = view.grab().toImage()
    assert pixels.pixelColor(600, 250) == QColor(255, 210, 65)
    assert pixels.pixelColor(200, 250) == QColor("#444444")
    assert pixels.pixelColor(600, 80) == QColor("#111111")
    view.resize(1000, 800)
    pixels = view.grab().toImage()
    assert pixels.pixelColor(750, 212) == QColor(255, 210, 65)
    view.set_frame(image(480, 640))
    view.set_pose(metrics(480, 640))
    pixels = view.grab().toImage()
    assert pixels.pixelColor(650, 200) == QColor(255, 210, 65)
    assert pixels.pixelColor(80, 200) == QColor("#111111")
    QTest.qWait(330)
    assert not view.groups
    assert view.grab().toImage().pixelColor(650, 200) == QColor("#444444")
    frame = metrics(480, 640)
    frame["t"] -= .31
    view.set_pose(frame)
    assert not view.groups
    view.close()


def test_predicted_missing_low_confidence_and_invalid_points_are_hidden():
    view = SkeletonVideoView()
    view.resize(800, 800)
    view.set_frame(image(640, 480))
    frame = metrics()
    frame["pose"][9, 2] = .1
    frame["pose"][10] = [700, 200, .9]
    frame["hands"] = [{"side": "left", "state": "bridged", "predicted": True,
                        "pts": np.tile([160., 120.], (21, 1))}]
    view.set_pose(frame)
    assert view.grab().toImage().pixelColor(600, 250) == QColor("#444444")
    frame["hands"][0].update(state="fresh", predicted=False)
    frame["t"] = time.monotonic()
    view.set_pose(frame)
    assert view.grab().toImage().pixelColor(600, 250) == QColor(50, 230, 115)
    frame["hands"] = []
    view.set_pose(frame)
    assert view.grab().toImage().pixelColor(600, 250) == QColor("#444444")
    view.close()


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def test_preview_continues_during_inference_and_only_latest_frame_is_processed():
    entered, release = Event(), Event()
    processed, selections, previews, results, errors = [], [], [], [], []

    class Camera:
        count = 0
        released = False

        def read(self):
            time.sleep(.01)
            self.count += 1
            return True, np.full((48, 64, 3), min(255, self.count), np.uint8)

        def release(self):
            self.released = True

    class Engine:
        closed = False

        def set_action_key(self, key, side):
            selections.append((key, side))

        def process(self, frame, timestamp, *, draw):
            assert draw is False
            processed.append(int(frame[0, 0, 0]))
            if len(processed) == 1:
                entered.set()
                assert release.wait(5)
            return frame, {"t": timestamp, "pose": None, "hands": [], "frame_size": (64, 48)}

        def close(self):
            self.closed = True

    camera, engine = Camera(), Engine()
    with patch("rehab_service.desktop.app.open_camera", return_value=camera), \
            patch("rehab_service.desktop.app.PoseEngine", return_value=engine):
        thread = CaptureThread()
        thread.set_action_key("fist", "right")
        thread.frame.connect(lambda frame: previews.append(frame))
        thread.metrics.connect(lambda frame: results.append(frame))
        thread.error.connect(lambda error: errors.append(error))
        thread.start()
        try:
            wait_for(entered.is_set)
            wait_for(lambda: len(previews) >= 6)
            assert len(processed) == 1 and not results
            newest = camera.count
            thread.set_action_key("rub", "left")
            release.set()
            wait_for(lambda: len(results) >= 2)
            assert processed[1] >= newest
            assert ("fist", "right") in selections and ("rub", "left") in selections
            assert results[-1]["desktop_generation"] == thread.selection[2]
            assert not errors
        finally:
            release.set()
            thread.stop()
        assert camera.released and engine.closed


def test_old_action_results_do_not_update_new_action_or_overlay():
    window = MainWindow()
    window.thread = CaptureThread()
    window.thread.set_action_key("rub", "right")
    frame = metrics()
    frame["desktop_generation"] = 0
    window.on_metrics(frame)
    assert not window.video.groups
    frame["desktop_generation"] = window.thread.selection[2]
    window.on_metrics(frame)
    assert window.video.groups
    window.close()


def test_camera_open_failure_releases_inference_without_waiting_for_a_frame():
    class Engine:
        closed = False

        def close(self):
            self.closed = True

    engine, errors = Engine(), []
    with patch("rehab_service.desktop.app.open_camera", return_value=None), \
            patch("rehab_service.desktop.app.PoseEngine", return_value=engine):
        thread = CaptureThread()
        thread.error.connect(lambda error: errors.append(error))
        thread.start()
        try:
            wait_for(lambda: not thread.isRunning() and bool(errors))
            assert "没找到可用的摄像头" in errors[0]
            assert engine.closed
        finally:
            thread.stop()


def test_inference_failure_releases_camera_and_models():
    class Camera:
        released = False

        def read(self):
            time.sleep(.01)
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            self.released = True

    class Engine:
        closed = False

        def process(self, *args, **kwargs):
            raise RuntimeError("synthetic inference failure")

        def close(self):
            self.closed = True

    camera, engine, errors = Camera(), Engine(), []
    with patch("rehab_service.desktop.app.open_camera", return_value=camera), \
            patch("rehab_service.desktop.app.PoseEngine", return_value=engine):
        thread = CaptureThread()
        thread.error.connect(lambda error: errors.append(error))
        thread.start()
        try:
            wait_for(lambda: not thread.isRunning() and bool(errors))
            assert "synthetic inference failure" in errors[0]
            assert camera.released and engine.closed
        finally:
            thread.stop()


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("PASS", name)
