"""Single-client WebRTC signaling, media and durable session lifecycle."""

import asyncio
import copy
import json
import logging
import time

from av import VideoFrame
from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import h264
from fastapi import APIRouter, HTTPException, Request

from rehab_service.storage.database import now
from rehab_service.services.auth import current_user, session_access
from rehab_service.services.training import TrainingSession
from rehab_service.recognition.inference import PoseEngine

router = APIRouter()
engine = None
active_client = None
model_error = None
model_task = None
active_cleanup = None
offer_lock = asyncio.Lock()

# aiortc 1.14.0 has no public per-sender bitrate setter. This sets only the
# initial H.264 target; RTCP congestion feedback can still reduce it.
REMOTE_VIDEO_START_BITRATE = 2_000_000
h264.DEFAULT_BITRATE = REMOTE_VIDEO_START_BITRATE


async def initialize_engine():
    global engine, model_error
    try:
        engine = await asyncio.to_thread(PoseEngine)
        model_error = None
    except Exception as exc:
        model_error = str(exc)
        logging.exception("模型加载失败，业务接口仍然可用")


async def interrupt_session(app, identity, reason="client_exit"):
    record = await asyncio.to_thread(app.state.store.session, identity)
    if app.state.active_session_id == identity and active_cleanup is not None:
        await active_cleanup(reason)
    elif record["status"] not in ("completed", "interrupted"):
        await asyncio.to_thread(app.state.store.finish_session, identity, record["summary"],
                                record["elapsed_seconds"], record["started_at"], reason)
    return await asyncio.to_thread(app.state.store.session, identity)


async def close_models():
    global engine
    if active_cleanup is not None:
        await active_cleanup("server_shutdown")
    if model_task is not None:
        await model_task
    if engine is not None:
        engine.close()
        engine = None


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


@router.post("/rtc/offer")
async def offer(request: Request):
    global active_client, active_cleanup
    app = request.app
    user = await asyncio.to_thread(current_user, request)
    if engine is None:
        raise HTTPException(503, "模型尚未加载")
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("type") != "offer" or not payload.get("sdp"):
            raise ValueError("需要 SDP offer")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 64:
            raise ValueError("需要已登录用户创建的 session_id，请升级客户端")
        record = await asyncio.to_thread(session_access, request, user, session_id)
        patient = dict(record["patient"], target_reps=record["target"])
        session = TrainingSession({"patient": patient, "action": record["action"]})
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    async with offer_lock:
        if active_client is not None:
            raise HTTPException(409, "服务端已有设备正在训练")
        if session_id:
            await asyncio.to_thread(app.state.store.claim_session, session_id)
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        active_client = pc
        app.state.active_session_id = session_id
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
    started_sent = False
    worker_task = None
    process_lock = asyncio.Lock()
    finish_lock = asyncio.Lock()
    finished = None
    started_at = None
    started_clock = None
    last_checkpoint = 0.0
    checkpoint_task = None

    def with_identity(data):
        if session_id:
            data["session_id"] = session_id
        return data

    def send(data):
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(with_identity(data), ensure_ascii=False))

    async def finish(reason):
        nonlocal finished
        async with finish_lock:
            if finished is not None:
                return finished
            elapsed = time.monotonic() - started_clock if started_clock is not None else 0.0
            summary = with_identity(session.summary())
            summary.update({"elapsed_seconds": elapsed, "started_at": started_at,
                            "ended_at": now(), "reason": reason,
                            "status": "completed" if reason == "manual" and started_at else "interrupted",
                            "target_reached": summary["count"] >= summary["target"]})
            if checkpoint_task is not None:
                await checkpoint_task
            if session_id:
                record = await asyncio.to_thread(app.state.store.finish_session, session_id,
                                                 summary, elapsed, started_at, reason)
                summary.update({"status": record["status"], "ended_at": record["ended_at"]})
            finished = summary
            return summary

    async def close_session(reason="disconnected"):
        nonlocal closing
        global active_client, active_cleanup
        if closing:
            return
        closing = True
        # Wait for the current inference and action update before taking a final snapshot.
        async with process_lock:
            try:
                await finish(reason)
            except Exception:
                logging.exception("训练记录保存失败")
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
        if worker_task is not None and not worker_task.done():
            try:
                await worker_task
            except Exception:
                pass
        try:
            await pc.close()
        finally:
            if active_client is pc:
                active_client = None
                active_cleanup = None
                app.state.active_session_id = None

    active_cleanup = close_session

    async def stop_and_summarize(reason):
        async with process_lock:
            try:
                send(await finish(reason))
            except Exception:
                send({"type": "error", "message": "结果保存失败，请检查服务端存储空间"})
        # Free the slot even when the client's acknowledgement is lost.
        await asyncio.sleep(3)
        await close_session(reason)

    async def check_login():
        while not closing:
            await asyncio.sleep(2)
            try:
                await asyncio.to_thread(current_user, request)
            except Exception:
                send({"type": "error", "message": "登录已失效，本次训练已停止"})
                await close_session("auth_expired")
                return

    async def read_video(track):
        nonlocal pending, dropped_input
        try:
            while not closing and not stopping:
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
        nonlocal pending, processed_frames, fps_started, processed_fps, worker_task
        nonlocal started_at, started_clock, last_checkpoint, checkpoint_task
        try:
            while not closing and not stopping:
                await frame_ready.wait()
                if closing or stopping:
                    return
                frame = pending
                pending = None
                frame_ready.clear()
                async with process_lock:
                    if closing or stopping:
                        return
                    started = time.monotonic()
                    if started_at is None:
                        started_at, started_clock = now(), started
                    image = frame.to_ndarray(format="bgr24")
                    worker_task = asyncio.create_task(asyncio.to_thread(engine.process, image))
                    annotated, metrics = await asyncio.shield(worker_task)
                    worker_task = None
                    result = session.update(metrics)
                    if session_id and started - last_checkpoint >= 1 and (checkpoint_task is None or checkpoint_task.done()):
                        if checkpoint_task is not None:
                            await checkpoint_task
                        snapshot = copy.deepcopy(session.summary())
                        checkpoint_task = asyncio.create_task(asyncio.to_thread(
                            app.state.store.checkpoint, session_id, snapshot, started - started_clock, started_at))
                        last_checkpoint = started
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
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            send({"type": "error", "message": "推理失败：" + str(exc)})
            await close_session("inference_error")

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
                command = json.loads(message)
                kind = command.get("type")
            except (ValueError, AttributeError):
                send({"type": "error", "message": "消息格式错误"})
                return
            if kind == "stop":
                reason = command.get("reason", "manual")
                if reason not in ("manual", "user_discomfort"):
                    send({"type": "error", "message": "结束原因错误"})
                    return
                if finished is not None:
                    send(finished)
                elif not stopping:
                    stopping = True
                    tasks.append(asyncio.create_task(stop_and_summarize(reason)))
            elif kind == "ack":
                if finished is not None:
                    asyncio.create_task(close_session("manual"))
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
        engine.reset()
        engine.set_action_key(session.action.key, session.ctx.affected)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type="offer"))
        if not any(transceiver.kind == "video" for transceiver in pc.getTransceivers()):
            raise ValueError("offer 中没有视频轨")
        # The Android client selects H.264 on phones and software VP8 on the
        # RK3588 robot; keep both codecs available for negotiation.
        codecs = RTCRtpSender.getCapabilities("video").codecs
        preferred = sorted(codecs, key=lambda codec: codec.mimeType.lower() != "video/h264")
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(preferred)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        tasks.extend((asyncio.create_task(connection_timeout()), asyncio.create_task(check_login())))
        return {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
    except Exception as exc:
        await close_session()
        raise HTTPException(400, "WebRTC 协商失败：" + str(exc)) from exc
