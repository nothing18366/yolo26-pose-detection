"""HTTP endpoints for training."""

import asyncio
from fastapi import APIRouter, Body, Request

from rehab_service.storage.database import require
from rehab_service.services.auth import current_user, patient_access, run_access, session_access

router = APIRouter(prefix="/api/v1")


@router.post("/training-runs", status_code=201)
def create_run(request: Request, payload: dict = Body(...)):
    user = current_user(request)
    patient_access(user, payload.get("patient_id"))
    return request.app.state.store.create_run(payload, user["role"])



@router.get("/training-runs")
def history(request: Request, patient_id: str, days: int = 0, offset: int = 0):
    patient_access(current_user(request), patient_id)
    return request.app.state.store.runs(patient_id, days, offset)



@router.get("/training-runs/{identity}")
def run(identity: str, request: Request):
    return run_access(request, current_user(request), identity)



@router.post("/training-runs/{identity}/finish")
def finish_run(identity: str, request: Request):
    run_access(request, current_user(request), identity)
    return request.app.state.store.finish_run(identity)



@router.put("/training-runs/{identity}/feedback/{phase}")
def feedback(identity: str, phase: str, request: Request, payload: dict = Body(...)):
    user = current_user(request)
    run_access(request, user, identity)
    return request.app.state.store.feedback(identity, phase, payload, user["id"])



@router.post("/sessions", status_code=201)
def create_session(request: Request, payload: dict = Body(...)):
    run_access(request, current_user(request), payload.get("run_id"))
    return request.app.state.store.create_session(payload)



@router.get("/sessions/{identity}")
def session(identity: str, request: Request):
    return session_access(request, current_user(request), identity)



@router.post("/sessions/{identity}/interrupt")
async def interrupt(identity: str, request: Request, payload: dict = Body(default={})):
    user = await asyncio.to_thread(current_user, request)
    await asyncio.to_thread(session_access, request, user, identity)
    reason = payload.get("reason", "client_exit")
    require(reason in ("client_exit", "user_discomfort"), "结束原因错误")
    return await request.app.state.interrupt_session(identity, reason)

