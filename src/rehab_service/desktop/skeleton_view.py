"""Local camera image and display-only skeleton, using the Android pose contract."""

import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from rehab_service.services.skeleton import pose_message

BODY_LINKS = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
              (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
              (1, 3), (2, 4), (3, 5), (4, 6))
HAND_LINKS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class SkeletonVideoView(QWidget):
    def __init__(self):
        super().__init__()
        self.setAccessibleName("本地摄像头与骨架画面")
        self.image = QImage()
        self.message = "正在打开摄像头…"
        self.groups = []
        self.pose_size = None
        self.expires_at = 0.0
        self.expiry = QTimer(self)
        self.expiry.setSingleShot(True)
        self.expiry.timeout.connect(self.clear_pose)

    def set_frame(self, image):
        self.image = image
        self.message = ""
        self.update()

    def set_message(self, message):
        self.message = message
        self.image = QImage()
        self.clear_pose()

    def clear_pose(self):
        self.expiry.stop()
        self.groups = []
        self.update()

    def set_pose(self, metrics):
        self.clear_pose()
        if "frame_size" not in metrics or time.monotonic() - metrics["t"] >= .3:
            return
        packet = pose_message(metrics, "qt", 0, metrics["t"] * 1000, time.monotonic() * 1000)
        self.pose_size = (packet["width"], packet["height"])
        if packet["body"] is not None:
            body = [point[:2] if point is not None and point[2] >= .25 else None
                    for point in packet["body"]]
            self.groups.append((QColor(255, 210, 65), body, BODY_LINKS))
        for hand in packet["hands"]:
            if hand["points"] is not None:
                color = QColor(50, 230, 115) if hand["side"] == "left" else QColor(75, 185, 255)
                self.groups.append((color, hand["points"], HAND_LINKS))
        self.expires_at = metrics["t"] + .3
        self.expiry.start(max(1, round((self.expires_at - time.monotonic()) * 1000)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self.image.isNull():
            painter.setPen(QColor("#eeeeee"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        left, top = (self.width() - width) / 2, (self.height() - height) / 2
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(QRectF(left, top, width, height), self.image)
        if time.monotonic() >= self.expires_at or self.pose_size != (self.image.width(), self.image.height()):
            return
        painter.setRenderHint(QPainter.Antialiasing)
        for color, points, links in self.groups:
            screen = [QPointF(left + (1 - p[0]) * width, top + p[1] * height)
                      if p is not None else None for p in points]
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            for start, end in links:
                if screen[start] is not None and screen[end] is not None:
                    painter.drawLine(screen[start], screen[end])
            for point in screen:
                if point is not None:
                    painter.drawEllipse(point, 2.5, 2.5)
