"""HTTP endpoints for auth."""

from fastapi import APIRouter, Body, Request

from rehab_service.services.auth import current_user

router = APIRouter(prefix="/api/v1")


@router.post("/auth/email-code")
def email_code(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.email_code(payload, request.client.host)



@router.post("/auth/email-login")
def email_login(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.email_login(payload, request.client.host)



@router.post("/auth/staff-login")
def staff_login(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.staff_login(payload, request.client.host)



@router.post("/auth/patient-login")
def patient_login(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.patient_login(payload, request.client.host)



@router.post("/auth/password-reset-code")
def password_reset_code(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.password_reset_code(payload, request.client.host)



@router.post("/auth/password-reset")
def password_reset(request: Request, payload: dict = Body(...)):
    return request.app.state.auth.password_reset(payload, request.client.host)



@router.get("/auth/me")
def me(request: Request):
    user = current_user(request)
    return {k: user[k] for k in ("id", "role", "patient_id", "expires")}



@router.post("/auth/logout")
def logout(request: Request):
    return request.app.state.auth.logout(current_user(request))

