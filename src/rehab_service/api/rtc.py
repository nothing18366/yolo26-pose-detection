"""Single-client WebRTC signaling, media and durable session lifecycle."""

import asyncio
import copy
import json
import logging
import os
import time
from pathlib import Path

from av import VideoFrame
from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import h264
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from rehab_service.storage.database import now
from rehab_service.services.auth import current_user, session_access, staff
from rehab_service.services.skeleton import pose_message
from rehab_service.services.training import TrainingSession
from rehab_service.recognition.inference import PoseEngine

router = APIRouter()
engine = None
active_client = None
model_error = None
model_task = None
active_cleanup = None
active_diagnostic = None
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


@router.get("/rtc/sessions/{identity}/frame.png")
async def diagnostic_frame(identity: str, request: Request):
    user = await asyncio.to_thread(current_user, request)
    staff(user)
    await asyncio.to_thread(session_access, request, user, identity)
    diagnostic = active_diagnostic
    if diagnostic is None or diagnostic["session_id"] != identity or diagnostic["frame"] is None:
        raise HTTPException(409, "当前会话未开启远端诊断或尚未收到图像")

    def encode(frame):
        import cv2
        ok, png = cv2.imencode(".png", frame.to_ndarray(format="bgr24"))
        if not ok:
            raise RuntimeError("PNG 编码失败")
        return png.tobytes()

    png = await asyncio.to_thread(encode, diagnostic["frame"])
    if active_diagnostic is not diagnostic:
        raise HTTPException(409, "诊断会话已结束")
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/rtc/offer")
async def offer(request: Request):
    global active_client, active_cleanup, active_diagnostic
    app = request.app
    user = await asyncio.to_thread(current_user, request)
    if engine is None:
        raise HTTPException(503, "模型尚未加载")
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("type") != "offer" or not payload.get("sdp"):
            raise ValueError("需要 SDP offer")
        preview_mode = payload.get("preview_mode", "keypoints")
        if preview_mode not in ("keypoints", "annotated", "diagnostic"):
            raise ValueError("不支持的显示模式")
        diagnostic = preview_mode == "diagnostic"
        if diagnostic:
            staff(user)
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
        active_diagnostic = {"session_id": session_id, "frame": None} if diagnostic else None
        app.state.active_session_id = session_id
    tasks = []
    channel = None
    pose_channel = None
    closing = False
    pending = None
    frame_ready = asyncio.Event()
    outgoing = LatestVideoTrack() if preview_mode != "keypoints" else None
    dropped_input = 0
    decoded_total = 0
    processed_total = 0
    skipped_pose = 0
    processed_frames = 0
    fps_started = time.monotonic()
    processed_fps = 0.0
    keypoint_frames = {"body": 0, "left_hand": 0, "right_hand": 0}
    keypoint_totals = dict(keypoint_frames)
    keypoint_fps = dict.fromkeys(keypoint_frames, 0.0)
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
    performance = None
    if os.environ.get("REHAB_PERFORMANCE_DIR"):
        try:
            directory = Path(os.environ["REHAB_PERFORMANCE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            performance = (directory / (session_id + ".jsonl")).open("a", buffering=1)
        except OSError:
            logging.exception("性能采样文件不可写，训练继续")

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
        global active_client, active_cleanup, active_diagnostic
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
                if performance is not None:
                    performance.close()
                active_client = None
                active_diagnostic = None
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
        nonlocal pending, dropped_input, decoded_total
        try:
            while not closing and not stopping:
                frame = await track.recv()
                decoded_total += 1
                if pending is not None:
                    dropped_input += 1
                pending = (frame, time.monotonic() * 1000)
                if diagnostic and active_diagnostic is not None:
                    active_diagnostic["frame"] = frame
                frame_ready.set()
        except asyncio.CancelledError:
            pass
        except Exception:
            await close_session()

    def infer(image):
        started_ms = time.monotonic() * 1000
        canvas, metrics = engine.process(image, draw=preview_mode == "annotated")
        return canvas, metrics, started_ms, time.monotonic() * 1000

    async def process_video():
        nonlocal pending, processed_frames, processed_total, fps_started, processed_fps, worker_task, skipped_pose, performance
        nonlocal started_at, started_clock, last_checkpoint, checkpoint_task
        try:
            while not closing and not stopping:
                await frame_ready.wait()
                if closing or stopping:
                    return
                frame, decoded_ms = pending
                pending = None
                frame_ready.clear()
                async with process_lock:
                    if closing or stopping:
                        return
                    started = time.monotonic()
                    if started_at is None:
                        started_at, started_clock = now(), started
                        fps_started = started
                    image = frame.to_ndarray(format="bgr24")
                    submitted_ms = time.monotonic() * 1000
                    worker_task = asyncio.create_task(asyncio.to_thread(infer, image))
                    canvas, metrics, inference_started_ms, inference_finished_ms = await asyncio.shield(worker_task)
                    worker_task = None
                    result = session.update(metrics)
                    if session_id and started - last_checkpoint >= 1 and (checkpoint_task is None or checkpoint_task.done()):
                        if checkpoint_task is not None:
                            await checkpoint_task
                        snapshot = copy.deepcopy(session.summary())
                        checkpoint_task = asyncio.create_task(asyncio.to_thread(
                            app.state.store.checkpoint, session_id, snapshot, started - started_clock, started_at))
                        last_checkpoint = started
                result_ready_ms = time.monotonic() * 1000
                timing = {
                    "input_wait_ms": started * 1000 - decoded_ms,
                    "image_conversion_ms": submitted_ms - started * 1000,
                    "worker_wait_ms": inference_started_ms - submitted_ms,
                    "inference_ms": inference_finished_ms - inference_started_ms,
                    "result_ms": result_ready_ms - inference_finished_ms,
                }
                processed_frames += 1
                processed_total += 1
                fresh_hands = {hand["side"] for hand in metrics.get("display_hands", metrics.get("hands", []))
                               if hand.get("state", "fresh") == "fresh" and not hand.get("predicted")}
                updates = {"body": metrics.get("pose") is not None and bool(metrics.get("pose_inferred")),
                           "left_hand": "left" in fresh_hands, "right_hand": "right" in fresh_hands}
                for name, fresh in updates.items():
                    keypoint_frames[name] += int(fresh)
                    keypoint_totals[name] += int(fresh)
                elapsed = time.monotonic() - fps_started
                if elapsed >= 1:
                    processed_fps = processed_frames / elapsed
                    for name in keypoint_frames:
                        keypoint_fps[name] = keypoint_frames[name] / elapsed
                        keypoint_frames[name] = 0
                    processed_frames = 0
                    fps_started = time.monotonic()
                result.update({
                    "server_total_ms": (time.monotonic() - started) * 1000,
                    "processed_fps": processed_fps,
                    "dropped_input": dropped_input,
                    "dropped_output": outgoing.dropped if outgoing is not None else 0,
                    "decoded_frames_total": decoded_total,
                    "processed_frames_total": processed_total,
                    "recognition_ms": metrics.get("recognition_ms"),
                    "drawing_ms": metrics.get("drawing_ms"),
                    "pose_inferred": bool(metrics.get("pose_inferred")),
                    "pose_inference_ms": metrics.get("pose_inference_ms"),
                    "hand_detection_ms": metrics.get("hand_detection_ms"),
                    "tracking_ms": metrics.get("tracking_ms"),
                    "hand_full_detections": metrics.get("hand_full_detections"),
                    "hand_crop_detections": metrics.get("hand_crop_detections"),
                    "timing": timing,
                    "preview_mode": preview_mode,
                    "skipped_pose": skipped_pose,
                })
                for name in keypoint_frames:
                    result[name + "_fps"] = keypoint_fps[name]
                    result[name + "_updates_total"] = keypoint_totals[name]
                if outgoing is not None:
                    output_frame = frame
                    if preview_mode == "annotated":
                        output_frame = VideoFrame.from_ndarray(canvas, format="bgr24")
                        output_frame.pts, output_frame.time_base = frame.pts, frame.time_base
                    outgoing.push(output_frame)
                if started_sent:
                    send(result)
                    if pose_channel is not None and pose_channel.readyState == "open":
                        if pose_channel.bufferedAmount == 0:
                            packet = pose_message(metrics, session_id, processed_total, decoded_ms,
                                                  time.monotonic() * 1000)
                            send_started_ms = time.monotonic() * 1000
                            packet["timing"] = dict(timing, send_started_at_ms=send_started_ms,
                                                  result_to_send_ms=send_started_ms - result_ready_ms)
                            pose_channel.send(json.dumps(packet, separators=(",", ":"), allow_nan=False))
                            timing["result_to_send_ms"] = send_started_ms - result_ready_ms
                            timing["pose_send_call_ms"] = time.monotonic() * 1000 - send_started_ms
                        else:
                            skipped_pose += 1
                if performance is not None:
                    try:
                        performance.write(json.dumps(dict(result, sampled_at=time.time(), seq=processed_total,
                            fresh_left_hand="left" in fresh_hands, fresh_right_hand="right" in fresh_hands)) + "\n")
                    except OSError:
                        logging.exception("性能采样写入失败，训练继续")
                        performance.close()
                        performance = None
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
        nonlocal channel, pose_channel, started_sent
        if datachannel.label == "pose":
            if preview_mode == "annotated":
                datachannel.close()
                return
            if datachannel.ordered or datachannel.maxRetransmits != 0:
                datachannel.close()
                tasks.append(asyncio.create_task(close_session("protocol_error")))
            else:
                pose_channel = datachannel
            return
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
                "preview_mode": preview_mode,
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
            if kind == "clock_ping":
                client_ms = command.get("client_ms")
                if isinstance(client_ms, (int, float)) and not isinstance(client_ms, bool) and 0 <= client_ms < 1e15:
                    send({"type": "clock_pong", "client_ms": client_ms, "server_ms": time.monotonic() * 1000})
            elif kind == "stop":
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
        if outgoing is not None:
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
