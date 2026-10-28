"""Pose serialization and diagnostic authorization at the HTTP boundary."""

import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.services.skeleton import pose_message
from rehab_service.api import rtc
from rehab_service.app import app
from test_rtc import FakeEngine


def test_pose_is_normalized_and_never_promotes_predicted_hands():
    pose = np.tile([320., 240., .9], (17, 1))
    pose[0, 0] = np.nan
    hands = [{"side": "left", "state": "bridged", "predicted": True, "pts": np.zeros((21, 2))},
             {"side": "right", "state": "fresh", "pts": np.tile([64., 48.], (21, 1))}]
    packet = pose_message({"pose": pose, "hands": hands, "frame_size": (640, 480), "t": 1., "pose_t": 1.},
                          "session", 12, 1000., 1050.)
    assert packet["body"][0] is None and packet["body"][1] == [.5, .5, .9]
    assert packet["hands"][0]["points"] is None
    assert packet["hands"][1]["points"][0] == [.1, .1]
    assert packet["hands"][1]["confidence"] is None
    json.dumps(packet, allow_nan=False)


def test_diagnostic_requires_staff_before_claiming_a_slot():
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/test.sqlite3"}), \
            patch.object(rtc, "engine", FakeEngine()), TestClient(app) as client:
        staff_token = client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
        client.headers["Authorization"] = "Bearer " + staff_token
        patient = client.post("/api/v1/patients", json={"name": "权限测试", "affected_side": "right", "forearm_cm": 25}).json()
        app.state.auth.save_patient_account(patient["id"], {"username": "pose-test", "password": "test-pass-123"})
        run = client.post("/api/v1/training-runs", json={"patient_id": patient["id"], "action": "fist", "mode": "single", "target_reps": 10}).json()
        client.put("/api/v1/training-runs/" + run["id"] + "/feedback/pre", json={"state": "skipped"})
        session = client.post("/api/v1/sessions", json={"run_id": run["id"], "item_index": 0}).json()
        token = client.post("/api/v1/auth/patient-login", json={"username": "pose-test", "password": "test-pass-123"}).json()["token"]
        client.headers["Authorization"] = "Bearer " + token
        response = client.post("/rtc/offer", json={"type": "offer", "sdp": "bad", "session_id": session["id"], "preview_mode": "diagnostic"})
        assert response.status_code == 403, response.text
        assert rtc.active_client is None and rtc.active_diagnostic is None
        assert app.state.store.session(session["id"])["status"] == "pending"
        assert client.get("/rtc/sessions/" + session["id"] + "/frame.png").status_code == 403
        client.headers.clear()
        assert client.get("/rtc/sessions/" + session["id"] + "/frame.png").status_code == 401
        client.headers["Authorization"] = "Bearer " + staff_token
        assert client.post("/rtc/offer", json={"type": "offer", "sdp": "bad", "session_id": session["id"], "preview_mode": "invalid"}).status_code == 400
        assert rtc.active_client is None


def test_pose_prefers_raw_display_points_and_rejects_predicted_display_hands():
    action_hand = {"side": "right", "state": "fresh", "pts": np.tile([64., 48.], (21, 1))}
    display_hand = {**action_hand, "pts": np.tile([320., 240.], (21, 1))}
    metrics = {"pose": None, "hands": [action_hand], "display_hands": [display_hand],
               "frame_size": (640, 480), "t": 1.}
    packet = pose_message(metrics, "session", 1, 1000., 1050.)
    assert packet["hands"][0]["points"][0] == [.5, .5]
    np.testing.assert_array_equal(action_hand["pts"][0], [64., 48.])
    for state, predicted in (("bridged", True), ("fresh", True), ("lost", False)):
        display_hand.update(state=state, predicted=predicted)
        assert pose_message(metrics, "session", 2, 1000., 1050.)["hands"][0]["points"] is None
    metrics["display_hands"] = []
    assert pose_message(metrics, "session", 3, 1000., 1050.)["hands"] == []


if __name__ == "__main__":
    test_pose_is_normalized_and_never_promotes_predicted_hands()
    test_pose_prefers_raw_display_points_and_rejects_predicted_display_hands()
    test_diagnostic_requires_staff_before_claiming_a_slot()
    print("PASS skeleton serialization and diagnostic permissions")
