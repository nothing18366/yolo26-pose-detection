"""Exercise durable records through real ICE, DataChannel and video tracks."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi.testclient import TestClient
from test_rtc import FakeEngine, TestVideoTrack
from rehab_service.api import rtc as server
from rehab_service.app import app


async def wait_for(events, kind):
    while True:
        item = await asyncio.wait_for(events.get(), 10)
        if item["type"] == kind:
            return item


async def exercise(client, action, interrupted=False, acknowledge=True, discomfort=False, revoked=False):
    response = client.post("/api/v1/patients", json={"name": "联调患者", "affected_side": "right", "forearm_cm": 25})
    patient = response.json()
    run = client.post("/api/v1/training-runs", json={"patient_id": patient["id"], "mode": "single", "action": action, "target_reps": 8}).json()
    assert client.put("/api/v1/training-runs/" + run["id"] + "/feedback/pre", json={"state": "skipped"}).status_code == 200
    session = client.post("/api/v1/sessions", json={"run_id": run["id"], "item_index": 0}).json()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    channel = pc.createDataChannel("rehab")
    events = asyncio.Queue()
    pc.addTrack(TestVideoTrack())

    @channel.on("message")
    def message(data):
        events.put_nowait(json.loads(data))

    try:
        await pc.setLocalDescription(await pc.createOffer())
        offer = {"type": "offer", "sdp": pc.localDescription.sdp, "session_id": session["id"]}
        response = await asyncio.to_thread(client.post, "/rtc/offer", json=offer)
        assert response.status_code == 200, response.text
        answer = response.json()
        await pc.setRemoteDescription(RTCSessionDescription(type="answer", sdp=answer["sdp"]))
        started = await wait_for(events, "started")
        assert started["session_id"] == session["id"]
        result = await wait_for(events, "result")
        assert result["session_id"] == session["id"]
        assert server.engine.action_key == action
        assert server.engine.affected_side == "right"
        assert (await asyncio.to_thread(client.post, "/rtc/offer", json=offer)).status_code == 409
        if revoked:
            assert (await asyncio.to_thread(client.post, "/api/v1/auth/logout")).status_code == 200
            event = await wait_for(events, "error")
            assert "登录已失效" in event["message"]
            client.headers["Authorization"] = "Bearer " + client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
        elif interrupted:
            response = await asyncio.to_thread(client.post, "/api/v1/sessions/" + session["id"] + "/interrupt")
            assert response.status_code == 200, response.text
            saved = response.json()
            assert saved["status"] == "interrupted" and saved["partial"]
            assert saved["summary"]["count"] == 0
        else:
            channel.send(json.dumps({"type": "stop", "reason": "user_discomfort" if discomfort else "manual"}))
            summary = await wait_for(events, "summary")
            assert summary["session_id"] == session["id"]
            saved = (await asyncio.to_thread(client.get, "/api/v1/sessions/" + session["id"])).json()
            assert saved["status"] == ("interrupted" if discomfort else "completed"), saved
            assert saved["elapsed_seconds"] >= saved["summary"]["total"]
            assert saved["summary"]["count"] == summary["count"]
            channel.send(json.dumps({"type": "stop"}))
            repeated = await wait_for(events, "summary")
            assert repeated == summary
            if acknowledge:
                channel.send(json.dumps({"type": "ack"}))
        for _ in range(45):
            if server.active_client is None:
                break
            await asyncio.sleep(0.1)
        assert server.active_client is None
        run = client.get("/api/v1/training-runs/" + run["id"]).json()
        assert run["status"] == ("unfinished" if discomfort else "interrupted" if interrupted or revoked else "completed")
        if revoked: assert run["end_reason"] == "auth_expired"
        if discomfort: assert run["end_reason"] == "user_discomfort"
        assert not run["sessions"][0]["target_reached"]
    finally:
        await pc.close()


def test_durable_rtc():
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/db.sqlite3"}):
        server.engine = FakeEngine()
        server.active_client = None
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
            for index, action in enumerate(("rub", "fist", "stretch", "cup", "abduction", "press")):
                asyncio.run(exercise(client, action, interrupted=action == "cup", acknowledge=action != "press"))
                print("PASS durable RTC", action, flush=True)
            asyncio.run(exercise(client, "cup", discomfort=True))
            print("PASS discomfort RTC", flush=True)
            asyncio.run(exercise(client, "cup", revoked=True))
            print("PASS revoked RTC", flush=True)


if __name__ == "__main__":
    test_durable_rtc()
