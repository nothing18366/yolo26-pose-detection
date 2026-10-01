"""WebRTC signaling and media test without loading the real models."""

import asyncio
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from av import VideoFrame
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inference
import main
import server


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

    def process(self, image):
        return image.copy(), {
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


async def test_webrtc(client):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    datachannel = pc.createDataChannel("rehab")
    events = asyncio.Queue()
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
        await asyncio.wait_for(received_video.wait(), 10)

        busy = await asyncio.to_thread(client.post, "/rtc/offer", json=payload)
        assert busy.status_code == 409
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
    finally:
        await pc.close()


def test_service():
    server.engine = FakeEngine()
    server.active_client = None
    assert main.PoseEngine is inference.PoseEngine
    assert server.PoseEngine is inference.PoseEngine
    with TestClient(server.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True and health["device"] == "test"
        assert client.post("/rtc/offer", json=offer_data("bad", action="unknown")).status_code == 400
        assert client.post("/rtc/offer", json=offer_data("bad")).status_code == 400
        asyncio.run(test_webrtc(client))


if __name__ == "__main__":
    test_service()
    print("PASS test_service")
