"""HTTP endpoints for teaching."""

from fastapi import APIRouter, Body, Request

from rehab_service.services.auth import current_user, staff

router = APIRouter(prefix="/api/v1")


@router.get("/actions")
def actions(request: Request):
    current_user(request)
    return request.app.state.teaching.catalog()



@router.patch("/actions/{key}/review")
def review(key: str, request: Request, payload: dict = Body(...)):
    user = current_user(request)
    staff(user)
    return request.app.state.teaching.review(key, payload, user["id"])

