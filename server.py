"""Single-client WebRTC service for Android rehabilitation training."""

import argparse
import asyncio
import json
import time

import uvicorn
from av import VideoFrame
from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import h264
from fastapi import FastAPI, HTTPException, Request

from actions import REGISTRY, ActionContext, FingerStretch
from inference import PoseEngine
from version import __version__

app = FastAPI(title="康复动作识别服务", version=__version__)
engine = None
active_client = None
ACTIONS = {cls.key: cls for cls in REGISTRY}

# aiortc 1.14.0 has no public per-sender bitrate setter. This sets only the
# initial H.264 target; RTCP congestion feedback can still reduce it.
REMOTE_VIDEO_START_BITRATE = 2_000_000
h264.DEFAULT_BITRATE = REMOTE_VIDEO_START_BITRATE


def action_detail(action):
    if isinstance(action, FingerStretch):
        return "背伸 %d 次 / 掌屈 %d 次" % (
            len(action.modes["ext"]), len(action.modes["flex"]))
    if action.key == "cup":
        return "端起 %d / 放下 %d" % (action.lift_count, action.place_count)
    if action.key == "abduction":
        return "拇指 %d / 食指 %d" % (
            len(action.modes["thumb"]), len(action.modes["index"]))
    if action.key == "press":
        return "目标A %d / 目标B %d" % (
            len(action.targets["a"]), len(action.targets["b"]))
    return "-"


class TrainingSession:
    def __init__(self, payload):
        patient = payload.get("patient") or {}
        side = patient.get("affected_side")
        action_key = payload.get("action")
        if side not in ("left", "right"):
            raise ValueError("患侧必须是 left 或 right")
        if action_key not in ACTIONS:
            raise ValueError("不支持的动作：%s" % action_key)
        forearm_cm = float(patient.get("forearm_cm", 0))
        target_reps = int(patient.get("target_reps", 0))
        if not 10.0 <= forearm_cm <= 45.0:
            raise ValueError("前臂长度必须在 10-45 cm")
        if not 1 <= target_reps <= 100:
            raise ValueError("目标次数必须在 1-100")
        self.ctx = ActionContext(
            str(patient.get("name") or "未填写"), side, forearm_cm, target_reps)
        cls = ACTIONS[action_key]
        self.action = cls(self.ctx)

    def update(self, frame):
        self.ctx.update_forearm(frame["pose"])
        self.action.update(frame)
        status = self.action.status()
        status.update({
            "type": "result",
            "detail": action_detail(self.action),
            "forearm_px": self.ctx.forearm_px,
            "affected_side": self.ctx.affected,
            "healthy_side": self.ctx.healthy,
            "hand_tracking": frame.get("hand_tracking", {}),
            "tracking_confidence": frame.get("tracking_confidence", 0.0),
            "processing_ms": frame.get("processing_ms", 0.0),
        })
        return status

    def summary(self):
        result = self.action.summary()
        result.update({"type": "summary", "detail": action_detail(self.action)})
        return result


@app.on_event("startup")
def load_models():
    global engine
    if engine is None:
        engine = PoseEngine()


@app.on_event("shutdown")
async def close_models():
    global engine
    if active_client is not None:
        await active_client.close()
    if engine is not None:
        engine.close()
        engine = None


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "loading",
        "model_loaded": engine is not None,
        "device": getattr(engine, "device", None),
        "version": app.version,
    }


class LatestVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frames = asyncio.Queue(maxsize=1)
        self.dropped = 0

    def push(self, frame):
        if self.frames.full():
            self.frames.get_nowait()
            self.dropped += 1
        self.frames.put_nowait(frame)

    async def recv(self):
        return await self.frames.get()


@app.post("/rtc/offer")
async def offer(request: Request):
    global active_client
    if engine is None:
        raise HTTPException(503, "模型尚未加载")
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("type") != "offer" or not payload.get("sdp"):
            raise ValueError("需要 SDP offer")
        session = TrainingSession(payload)
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if active_client is not None:
        raise HTTPException(409, "服务端已有设备正在训练")

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    active_client = pc
    engine.reset()
    engine.set_action_key(session.action.key, session.ctx.affected)
    tasks = []
    channel = None
    closing = False
    pending = None
    frame_ready = asyncio.Event()
    outgoing = LatestVideoTrack()
    dropped_input = 0
    processed_frames = 0
    fps_started = time.monotonic()
    processed_fps = 0.0
    stopping = False
    processing = False
    started_sent = False
    worker_task = None

    def send(data):
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(data, ensure_ascii=False))

    async def close_session():
        nonlocal closing
        global active_client
        if closing:
            return
        closing = True
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
        if worker_task is not None and not worker_task.done():
            try:
                await worker_task
            except Exception:
                pass
        await pc.close()
        if active_client is pc:
            active_client = None

    async def read_video(track):
        nonlocal pending, dropped_input
        try:
            while True:
                frame = await track.recv()
                if pending is not None:
                    dropped_input += 1
                pending = frame
                frame_ready.set()
        except asyncio.CancelledError:
            pass
        except Exception:
            await close_session()

    async def process_video():
        nonlocal pending, processed_frames, fps_started, processed_fps, processing, worker_task
        try:
            while True:
                await frame_ready.wait()
                frame = pending
                pending = None
                frame_ready.clear()
                processing = True
                started = time.monotonic()
                image = frame.to_ndarray(format="bgr24")
                worker_task = asyncio.create_task(asyncio.to_thread(engine.process, image))
                annotated, metrics = await asyncio.shield(worker_task)
                worker_task = None
                result = session.update(metrics)
                processed_frames += 1
                elapsed = time.monotonic() - fps_started
                if elapsed >= 1:
                    processed_fps = processed_frames / elapsed
                    processed_frames = 0
                    fps_started = time.monotonic()
                result.update({
                    "server_total_ms": (time.monotonic() - started) * 1000,
                    "processed_fps": processed_fps,
                    "dropped_input": dropped_input,
                    "dropped_output": outgoing.dropped,
                })
                annotated_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
                annotated_frame.pts = frame.pts
                annotated_frame.time_base = frame.time_base
                outgoing.push(annotated_frame)
                if started_sent:
                    send(result)
                processing = False
                if stopping:
                    send(session.summary())
                    return
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            send({"type": "error", "message": "推理失败：" + str(exc)})
            await close_session()

    async def connection_timeout():
        await asyncio.sleep(15)
        if pc.connectionState != "connected":
            await close_session()

    @pc.on("datachannel")
    def on_datachannel(datachannel):
        nonlocal channel, started_sent
        if datachannel.label != "rehab":
            return
        channel = datachannel

        @channel.on("open")
        def on_open():
            nonlocal started_sent
            if started_sent:
                return
            started_sent = True
            send({
                "type": "started",
                "action": session.action.key,
                "name": session.action.name,
                "target": session.ctx.target_reps,
            })

        if channel.readyState == "open":
            on_open()

        @channel.on("message")
        def on_message(message):
            nonlocal stopping
            if not isinstance(message, str):
                return
            try:
                kind = json.loads(message).get("type")
            except (ValueError, AttributeError):
                send({"type": "error", "message": "消息格式错误"})
                return
            if kind == "stop":
                stopping = True
                if not processing:
                    send(session.summary())
            elif kind == "ack":
                asyncio.create_task(close_session())
            else:
                send({"type": "error", "message": "不支持的消息类型"})

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return
        pc.addTrack(outgoing)
        tasks.extend((asyncio.create_task(read_video(track)), asyncio.create_task(process_video())))

        @track.on("ended")
        def on_ended():
            asyncio.create_task(close_session())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ("failed", "disconnected", "closed"):
            await close_session()

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type="offer"))
        if not any(transceiver.kind == "video" for transceiver in pc.getTransceivers()):
            raise ValueError("offer 中没有视频轨")
        codecs = RTCRtpSender.getCapabilities("video").codecs
        preferred = sorted(codecs, key=lambda codec: codec.mimeType.lower() != "video/h264")
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(preferred)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        tasks.append(asyncio.create_task(connection_timeout()))
        return {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
    except Exception as exc:
        await close_session()
        raise HTTPException(400, "WebRTC 协商失败：" + str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description="康复动作识别服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
