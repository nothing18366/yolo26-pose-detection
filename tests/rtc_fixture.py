"""Disposable robot-test service: temporary database, no email, fixed port 18121."""

import asyncio
import os
from pathlib import Path
import sys
import tempfile

import uvicorn
from fastapi import HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.api.rtc import LatestVideoTrack
from rehab_service.app import app
from rehab_service.services.auth import current_user, staff

paused = False
push_frame = LatestVideoTrack.push


def push(self, frame):
    if not paused:
        push_frame(self, frame)


LatestVideoTrack.push = push


@app.post("/test/return-video/{command}")
async def return_video(command: str, request: Request):
    global paused
    staff(await asyncio.to_thread(current_user, request))
    if command not in ("pause", "resume"):
        raise HTTPException(400, "Invalid test command")
    paused = command == "pause"
    return {"paused": paused}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="rehab-rtc-") as directory:
        os.environ["REHAB_DB"] = str(Path(directory) / "test.sqlite3")
        for key in tuple(os.environ):
            if key.startswith("SMTP_"):
                del os.environ[key]
        uvicorn.run(app, host="0.0.0.0", port=18121)
