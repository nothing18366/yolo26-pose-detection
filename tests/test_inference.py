"""Current-frame rendering and action-aware hand detection regressions."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import PoseEngine, detect_hands_by_half


class FakeResult:
    class keypoints:
        data = []

    def plot(self, img):
        return img


class FakeModel:
    calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        return [FakeResult()]


def test_cached_pose_uses_current_background():
    engine = PoseEngine.__new__(PoseEngine)
    engine.device = "cpu"
    engine.model = FakeModel()
    engine.hands = None
    engine.guidance_rois = {}
    engine.reset()
    with patch("inference.detect_hands_by_half", return_value=[]):
        first, _ = engine.process(np.zeros((48, 64, 3), dtype=np.uint8))
        second, _ = engine.process(np.full((48, 64, 3), 255, dtype=np.uint8))
    assert engine.model.calls == 1
    assert first[0, 0, 0] == 0
    assert second[0, 0, 0] == 255


def test_duplicate_hand_triggers_forearm_crop():
    class FakeLandmarker:
        def detect(self, image):
            h, w = image.numpy_view().shape[:2]

            def hand(x, y):
                return [SimpleNamespace(x=x / w, y=(y + i * 6) / h, z=0.0)
                        for i in range(21)]

            if (h, w) == (1280, 720):
                hands = [hand(420, 820), hand(425, 825)]
            else:
                hands = [hand(220, 233)]
            return SimpleNamespace(hand_landmarks=hands, handedness=[
                [SimpleNamespace(category_name="Left")] for _ in hands])

    pose = np.zeros((17, 3))
    pose[8] = [160, 500, 0.99]
    pose[10] = [400, 850, 0.99]
    hands = detect_hands_by_half(FakeLandmarker(), np.zeros((1280, 720, 3), dtype=np.uint8), pose=pose)
    assert len(hands) == 2
    assert np.linalg.norm(hands[0]["pts"][0] - hands[1]["pts"][0]) > 100


def test_abduction_crop_keeps_body_side_and_world_landmarks():
    class FakeLandmarker:
        def detect(self, image):
            points = [SimpleNamespace(x=.5, y=.5 - i * .005, z=0.0) for i in range(21)]
            world = [SimpleNamespace(x=i * .001, y=i * .002, z=i * .003) for i in range(21)]
            return SimpleNamespace(hand_landmarks=[points], hand_world_landmarks=[world],
                                   handedness=[[SimpleNamespace(category_name="Left")]])

    pose = np.zeros((17, 3))
    pose[8] = [200, 600, .99]
    pose[10] = [300, 800, .99]
    hands = detect_hands_by_half(FakeLandmarker(), np.zeros((1280, 720, 3), dtype=np.uint8),
                                pose=pose, wrist_crops=True)
    assert len(hands) == 1 and hands[0]["side"] == "right"
    assert hands[0]["world"].shape == (21, 3)


class CountingLandmarker:
    def __init__(self, categories, position=.25):
        self.categories = categories
        self.position = position
        self.calls = 0

    def detect(self, image):
        category = self.categories[min(self.calls, len(self.categories) - 1)]
        self.calls += 1
        points = [SimpleNamespace(x=self.position, y=self.position + i * .005, z=0.0) for i in range(21)]
        return SimpleNamespace(hand_landmarks=[points] if category else [],
                               handedness=[[SimpleNamespace(category_name=category)]] if category else [])


def test_single_hand_actions_do_not_search_for_healthy_hand():
    for side, category in (("left", "Right"), ("right", "Left")):
        detector = CountingLandmarker([category])
        hands = detect_hands_by_half(detector, np.zeros((480, 640, 3), dtype=np.uint8),
                                     guidance_rois={"extra": (.4, .4, .9, .9)}, required_side=side)
        assert len(hands) == 1 and detector.calls == 1, (side, detector.calls)


def test_healthy_hand_alone_does_not_stop_affected_hand_search():
    # Category Left denotes the body-right hand for the non-mirrored input.
    detector = CountingLandmarker(["Left", "Right"])
    hands = detect_hands_by_half(detector, np.zeros((480, 640, 3), dtype=np.uint8),
                                 required_side="left")
    assert len(hands) == 2 and detector.calls == 2


def test_missing_affected_hand_is_not_invented():
    from tracking import HandTracker

    detector = CountingLandmarker(["Left", None])
    hands = detect_hands_by_half(detector, np.zeros((480, 640, 3), dtype=np.uint8),
                                 required_side="left")
    assert detector.calls == 3 and len(hands) == 1
    assert "left" not in HandTracker().assign(hands)


def test_bilateral_actions_still_search_for_second_hand():
    detector = CountingLandmarker(["Left", "Right"])
    hands = detect_hands_by_half(detector, np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(hands) == 2 and detector.calls == 2


def test_abduction_does_not_detect_healthy_wrist_crop():
    detector = CountingLandmarker(["Left"], position=.5)
    pose = np.zeros((17, 3))
    pose[7], pose[9] = [120, 160, .99], [160, 260, .99]
    pose[8], pose[10] = [500, 160, .99], [460, 260, .99]
    hands = detect_hands_by_half(detector, np.zeros((480, 640, 3), dtype=np.uint8),
                                 pose=pose, wrist_crops=True, required_side="right")
    assert detector.calls == 1 and len(hands) == 1 and hands[0]["side"] == "right"


def test_all_actions_request_only_the_hands_they_use():
    for side in ("left", "right"):
        for key in ("cup", "press", "abduction", "rub", "fist", "stretch"):
            engine = PoseEngine.__new__(PoseEngine)
            engine.device, engine.model, engine.hands = "cpu", FakeModel(), None
            engine.reset()
            engine.set_action_key(key, side)
            with patch("inference.detect_hands_by_half", return_value=[]) as detect:
                engine.process(np.zeros((48, 64, 3), dtype=np.uint8))
            assert detect.call_args.kwargs["required_side"] == (
                side if key in ("cup", "press", "abduction") else None), (key, side)


def test_qt_passes_affected_side_to_shared_engine():
    from main import CaptureThread

    thread = CaptureThread()
    thread.engine = SimpleNamespace(set_action_key=lambda key, side: calls.append((key, side)))
    calls = []
    for key in ("cup", "press", "abduction", "rub", "fist", "stretch"):
        thread.set_action_key(key, "right")
        assert calls[-1] == (key, "right") and thread.affected_side == "right"


def test_hand_assignment_does_not_advance_tracking_state():
    from tracking import HandTracker

    tracker = HandTracker()
    hand = {"cat": "Left", "pts": np.zeros((21, 2))}
    tracker.update([hand], 1.0)
    previous = tracker.last["right"]
    assert tracker.assign([hand])["right"] is hand
    assert tracker.last["right"] is previous and previous["_t"] == 1.0


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("PASS", name)
