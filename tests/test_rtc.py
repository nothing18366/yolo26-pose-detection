"""WebRTC signaling and media test without loading the real models."""

import asyncio
import json
import os
import tempfile
from unittest.mock import patch
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from av import VideoFrame
from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import rehab_service.recognition.inference as inference
from rehab_service.api import rtc as server
from rehab_service.app import app


class FakeEngine:
    device = "test"

    def __init__(self):
        self.action_key = None
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def set_action_key(self, key, affected_side=None):
        self.action_key = key
        self.affected_side = affected_side

    def process(self, image, *, draw=True):
        self.last_draw = draw
        return np.full_like(image, 255), {
            "t": monotonic(),
            "pose": None,
            "hands": [],
            "frame_size": (image.shape[1], image.shape[0]),
            "forearm_axis": None,
            "hand_tracking": {},
            "tracking_confidence": 0.0,
            "processing_ms": 1.25,
        }

    def close(self):
        pass


class TestVideoTrack(VideoStreamTrack):
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class LandmarkEngine(FakeEngine):
    def reset(self):
        super().reset()
        self.frames = 0

    def process(self, image, *, draw=True):
        output, metrics = super().process(image, draw=draw)
        self.frames += 1
        refreshed = self.frames % 2 == 1
        fresh_hand = self.frames % 3 == 1
        metrics.update({"pose": np.tile([160., 120., .9], (17, 1)),
                        "pose_inferred": refreshed, "pose_t": metrics["t"] if refreshed else metrics["t"] - 1,
                        "pose_inference_ms": .2 if refreshed else 0., "hand_detection_ms": .6, "tracking_ms": .1,
                        # Keep display fixtures out of the fake action's input.
                        "display_hands": [
                            {"side": "left", "state": "fresh" if fresh_hand else "bridged",
                             "predicted": not fresh_hand, "pts": np.tile([64., 48.], (21, 1))},
                            {"side": "right", "state": "fresh", "predicted": True,
                             "pts": np.tile([64., 48.], (21, 1))}]})
        return output, metrics


def offer_data(sdp, action="fist"):
    return {
        "type": "offer",
        "sdp": sdp,
        "patient": {
            "name": "患者A",
            "affected_side": "left",
            "forearm_cm": 25,
            "target_reps": 3,
        },
        "action": action,
        "mode": "auto",
    }


async def test_webrtc(client, diagnostic=False):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    datachannel = pc.createDataChannel("rehab")
    events = asyncio.Queue()
    pose = pc.createDataChannel("pose", ordered=False, maxRetransmits=0)
    poses = asyncio.Queue()
    pose.on("message", lambda message: poses.put_nowait(json.loads(message)))
    received_video = asyncio.Event()
    pc.addTrack(TestVideoTrack())

    @datachannel.on("message")
    def on_message(message):
        events.put_nowait(json.loads(message))

    @pc.on("track")
    def on_track(track):
        async def read_one():
            frame = await track.recv()
            assert frame.width == 320 and frame.height == 240
            received_video.set()
        asyncio.create_task(read_one())

    try:
        await pc.setLocalDescription(await pc.createOffer())
        payload = offer_data(pc.localDescription.sdp)
        patient = client.post("/api/v1/patients", json={"name": "RTC测试", "affected_side": "left", "forearm_cm": 25}).json()
        run = client.post("/api/v1/training-runs", json={"patient_id": patient["id"], "mode": "single", "action": "fist", "target_reps": 3}).json()
        client.put("/api/v1/training-runs/" + run["id"] + "/feedback/pre", json={"state": "skipped"})
        payload["session_id"] = client.post("/api/v1/sessions", json={"run_id": run["id"], "item_index": 0}).json()["id"]
        if diagnostic:
            payload["preview_mode"] = "diagnostic"
        response = await asyncio.to_thread(client.post, "/rtc/offer", json=payload)
        assert response.status_code == 200, response.text
        assert server.engine.action_key == "fist" and server.engine.affected_side == "left"
        answer = response.json()
        await pc.setRemoteDescription(RTCSessionDescription(
            type=answer["type"], sdp=answer["sdp"]))
        started = await asyncio.wait_for(events.get(), 10)
        assert started["type"] == "started", started
        result = await asyncio.wait_for(events.get(), 10)
        assert result["type"] == "result"
        assert result["name"] == "动作2 握拳伸展"
        assert result["processing_ms"] == 1.25
        while result["processed_fps"] == 0:
            result = await asyncio.wait_for(events.get(), 10)
        count = result["processed_frames_total"]
        assert result["body_updates_total"] == (count + 1) // 2
        assert result["left_hand_updates_total"] == (count + 2) // 3
        assert result["right_hand_updates_total"] == 0
        assert 0 < result["left_hand_fps"] < result["body_fps"] < result["processed_fps"]
        assert result["right_hand_fps"] == 0
        assert result["hand_detection_ms"] == .6 and result["tracking_ms"] == .1
        packet = await asyncio.wait_for(poses.get(), 10)
        assert packet["session_id"] == payload["session_id"] and packet["seq"] > 0
        assert packet["width"] == 320 and packet["height"] == 240
        assert packet["completed_at_ms"] >= packet["decoded_at_ms"]
        assert server.engine.last_draw is False
        path = "/rtc/sessions/" + payload["session_id"] + "/frame.png"
        if diagnostic:
            await asyncio.wait_for(received_video.wait(), 10)
            png = await asyncio.to_thread(client.get, path)
            assert png.status_code == 200 and png.content.startswith(b"\x89PNG\r\n\x1a\n")
            assert png.headers["cache-control"] == "no-store"
            import cv2
            assert not cv2.imdecode(np.frombuffer(png.content, dtype=np.uint8), cv2.IMREAD_COLOR).any()
        else:
            assert not received_video.is_set()
            assert all(sender.track is None for sender in server.active_client.getSenders())
            assert server.active_diagnostic is None
            assert (await asyncio.to_thread(client.get, path)).status_code == 409
        datachannel.send(json.dumps({"type": "clock_ping", "client_ms": 12345}))
        while True:
            pong = await asyncio.wait_for(events.get(), 10)
            if pong["type"] == "clock_pong":
                assert pong["client_ms"] == 12345 and pong["server_ms"] > 0
                break

        busy = await asyncio.to_thread(client.post, "/rtc/offer", json=payload)
        assert busy.status_code == 409
        buffered = RTCDataChannel.bufferedAmount
        # Simulate only pose congestion; reliable stop and summary must still complete.
        with patch.object(RTCDataChannel, "bufferedAmount", property(
                lambda channel: 16384 if channel.label == "pose" else buffered.fget(channel))):
            async def wait_for_skip():
                while True:
                    update = await events.get()
                    if update.get("skipped_pose", 0) > 0:
                        return
            await asyncio.wait_for(wait_for_skip(), 10)
            datachannel.send(json.dumps({"type": "stop"}))
            while True:
                summary = await asyncio.wait_for(events.get(), 10)
                if summary["type"] == "summary":
                    break
        assert summary["count"] == 0
        datachannel.send(json.dumps({"type": "ack"}))
        for _ in range(20):
            if server.active_client is None:
                break
            await asyncio.sleep(0.1)
        assert server.active_client is None
        assert (await asyncio.to_thread(client.get, path)).status_code == 409
    finally:
        await pc.close()


def test_service():
    server.engine = LandmarkEngine()
    server.active_client = None
    assert server.PoseEngine is inference.PoseEngine
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/db.sqlite3"}), TestClient(app) as client:
        assert client.post("/rtc/offer", json=offer_data("bad")).status_code == 401
        client.headers["Authorization"] = "Bearer " + client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
        health = client.get("/health").json()
        assert health["model_loaded"] is True and health["device"] == "test"
        assert client.post("/rtc/offer", json=offer_data("bad", action="unknown")).status_code == 400
        assert client.post("/rtc/offer", json=offer_data("bad")).status_code == 400
        asyncio.run(test_webrtc(client))
        asyncio.run(test_webrtc(client, diagnostic=True))


if __name__ == "__main__":
    test_service()
    print("PASS test_service")
