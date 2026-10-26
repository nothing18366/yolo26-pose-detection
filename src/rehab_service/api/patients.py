"""HTTP endpoints for patients."""

from fastapi import APIRouter, Body, Request

from rehab_service.services.auth import current_user, staff, patient_access

router = APIRouter(prefix="/api/v1")


@router.get("/patients")
def patients(request: Request, q: str = "", archived: bool = False):
    staff(current_user(request))
    return request.app.state.store.patients(q, archived)



@router.post("/patients", status_code=201)
def create_patient(request: Request, payload: dict = Body(...)):
    staff(current_user(request))
    return request.app.state.store.save_patient(payload)



@router.get("/patients/{identity}")
def patient(identity: str, request: Request):
    patient_access(current_user(request), identity)
    return request.app.state.store.patient(identity)



@router.patch("/patients/{identity}")
def update_patient(identity: str, request: Request, payload: dict = Body(...)):
    staff(current_user(request))
    return request.app.state.store.save_patient(payload, identity)



@router.get("/patients/{identity}/plan")
def plan(identity: str, request: Request):
    patient_access(current_user(request), identity)
    return request.app.state.store.plan(identity)



@router.get("/patients/{identity}/account")
def patient_account(identity: str, request: Request):
    staff(current_user(request))
    return request.app.state.auth.patient_account(identity)



@router.put("/patients/{identity}/account")
def save_patient_account(identity: str, request: Request, payload: dict = Body(...)):
    staff(current_user(request))
    return request.app.state.auth.save_patient_account(identity, payload)



@router.put("/patients/{identity}/plan")
def save_plan(identity: str, request: Request, payload: dict = Body(...)):
    staff(current_user(request))
    return request.app.state.store.save_plan(identity, payload)



@router.get("/patients/{identity}/overview")
def overview(identity: str, request: Request, utc_offset_minutes: int = 0):
    patient_access(current_user(request), identity)
    return request.app.state.store.overview(identity, 480)



@router.get("/patients/{identity}/calendar")
def calendar(identity: str, request: Request, week_start: str = None):
    patient_access(current_user(request), identity)
    return request.app.state.store.calendar(identity, week_start)



@router.get("/patients/{identity}/trends")
def trends(identity: str, request: Request, action: str, start: str, end: str,
           mode: str = None, revision: int = None):
    patient_access(current_user(request), identity)
    return request.app.state.store.trends(identity, action, start, end, mode, revision)

