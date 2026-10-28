"""FastAPI application assembly and command-line entry point."""

import argparse
import asyncio
from functools import partial
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rehab_service.api import auth, patients, rtc, teaching, training
from rehab_service.services.auth import AuthService
from rehab_service.services.teaching import TeachingService
from rehab_service.settings import DATA_DIR, MEDIA_DIR
from rehab_service.storage.database import BusinessError, RehabStore
from rehab_service.version import __version__

app = FastAPI(title="康复动作识别服务", version=__version__)
for router in (auth.router, patients.router, training.router, teaching.router, rtc.router):
    app.include_router(router)
app.mount("/media", StaticFiles(directory=MEDIA_DIR, check_dir=False), name="media")


@app.exception_handler(BusinessError)
async def business_error(request, exc):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.on_event("startup")
async def load_models():
    app.state.store = await asyncio.to_thread(RehabStore, os.environ.get("REHAB_DB", str(DATA_DIR / "rehab.sqlite3")))
    app.state.auth = await asyncio.to_thread(AuthService, app.state.store)
    app.state.teaching = await asyncio.to_thread(TeachingService, app.state.store)
    app.state.store.teaching_versions = {i["key"]: i["content_version"] for i in app.state.teaching.items}
    app.state.active_session_id = None
    app.state.interrupt_session = partial(rtc.interrupt_session, app)
    await asyncio.to_thread(app.state.store.recover)
    rtc.offer_lock = asyncio.Lock()
    if rtc.engine is None:
        rtc.model_task = asyncio.create_task(rtc.initialize_engine())


@app.on_event("shutdown")
async def close_models():
    await rtc.close_models()


@app.get("/health")
def health():
    return {
        "status": "ok" if rtc.engine is not None else "error" if rtc.model_error else "loading",
        "model_loaded": rtc.engine is not None,
        "device": getattr(rtc.engine, "device", None),
        "version": app.version,
        "model_error": rtc.model_error,
        "busy": rtc.active_client is not None,
        "business_api": "v1",
        "auth_required": True,
        "data_schema": 3,
        "capabilities": {"local_skeleton_v1": True, "diagnostic_video": True,
                         "diagnostic_png": True, "clock_sync_v1": True},
    }


def main():
    parser = argparse.ArgumentParser(description="康复动作识别服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
