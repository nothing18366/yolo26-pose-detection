"""Business contracts and recovery, without real models or camera hardware."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.storage.database import BusinessError, RehabStore, now


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = RehabStore(Path(self.directory.name) / "rehab.sqlite3")
        self.patient = self.store.save_patient({"name": "同名患者", "affected_side": "left", "forearm_cm": 25})
        self.other = self.store.save_patient({"name": "同名患者", "affected_side": "right", "forearm_cm": 27})

    def tearDown(self):
        self.directory.cleanup()

    def run_one(self, target=10):
        run = self.store.create_run({"patient_id": self.patient["id"], "mode": "single", "action": "cup", "target_reps": target})
        self.store.feedback(run["id"], "pre", {"state": "skipped"}, "test")
        return run

    def test_validation_and_unique_patients(self):
        self.assertNotEqual(self.patient["code"], self.other["code"])
        self.assertEqual(self.store.patients(self.patient["code"])[0]["id"], self.patient["id"])
        for values in ({"name": ""}, {"forearm_cm": float("nan")}, {"affected_side": "both"}):
            with self.assertRaises(BusinessError):
                self.store.save_patient(values, self.patient["id"])
        with self.assertRaises(BusinessError):
            self.store.save_plan(self.patient["id"], {"items": [{"action": "cup", "target_reps": 3}] * 2})

    def test_snapshot_isolation_and_archive(self):
        run = self.run_one()
        self.store.save_patient({"name": "改名", "affected_side": "right", "archived": True}, self.patient["id"])
        self.assertEqual(self.store.run(run["id"])["patient"]["name"], "同名患者")
        self.assertEqual(self.store.run(run["id"])["patient"]["affected_side"], "left")
        self.assertEqual(self.store.runs(self.other["id"]), [])
        self.assertEqual(len(self.store.runs(self.patient["id"])), 1)
        self.assertEqual(len(self.store.patients(archived=True)), 1)
        with self.assertRaises(BusinessError):
            self.run_one()

    def test_plan_order_and_revision_snapshot(self):
        items = [{"action": "cup", "target_reps": 10}, {"action": "press", "target_reps": 8}]
        self.store.save_plan(self.patient["id"], {"items": items})
        run = self.store.create_run({"patient_id": self.patient["id"], "mode": "plan"})
        self.store.feedback(run["id"], "pre", {"state": "skipped"}, "test")
        self.store.save_plan(self.patient["id"], {"items": list(reversed(items))})
        self.assertEqual(self.store.run(run["id"])["items"], items)
        self.assertEqual(run["plan_revision"], 1)
        with self.assertRaises(BusinessError):
            self.store.create_session({"run_id": run["id"], "item_index": 1})
        for index in range(2):
            session = self.store.create_session({"run_id": run["id"], "item_index": index})
            self.store.claim_session(session["id"])
            self.store.finish_session(session["id"], {"count": 1, "target": items[index]["target_reps"]}, 9, now())
        self.assertEqual(self.store.run(run["id"])["status"], "completed")

    def test_finalization_is_idempotent_and_time_is_separate(self):
        run = self.run_one()
        session = self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.claim_session(session["id"])
        with self.assertRaises(BusinessError):
            self.store.claim_session(session["id"])
        summary = {"count": 2, "target": 10, "passed": 1, "total": 3.5, "avg": 1.75}
        self.store.checkpoint(session["id"], summary, 10, now())
        self.store.finish_session(session["id"], summary, 13, now())
        self.store.finish_session(session["id"], {"count": 99}, 999, now(), "disconnected")
        result = self.store.session(session["id"])
        self.assertEqual(result["elapsed_seconds"], 13)
        self.assertEqual(result["summary"]["total"], 3.5)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["target_reached"])
        self.assertEqual(self.store.overview(self.patient["id"])["completed_sessions"], 1)

    def test_restart_keeps_partial_checkpoint_and_no_resume(self):
        run = self.run_one()
        session = self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.claim_session(session["id"])
        self.store.checkpoint(session["id"], {"count": 3, "target": 10}, 7, now())
        reopened = RehabStore(self.store.path)
        reopened.recover()
        result = reopened.session(session["id"])
        self.assertTrue(result["partial"])
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(reopened.run(run["id"])["status"], "interrupted")
        with self.assertRaises(BusinessError):
            reopened.create_session({"run_id": run["id"], "item_index": 0})

    def test_busy_profile_and_plan_cannot_change(self):
        run = self.run_one()
        session = self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.claim_session(session["id"])
        with self.assertRaises(BusinessError):
            self.store.save_patient({"name": "另一个人"}, self.patient["id"])
        with self.assertRaises(BusinessError):
            self.store.save_plan(self.patient["id"], {"items": [{"action": "fist", "target_reps": 10}]})

    def test_backup_and_early_exit(self):
        run = self.run_one()
        self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.finish_run(run["id"])
        self.assertEqual(self.store.run(run["id"])["status"], "unfinished")
        destination = Path(self.directory.name) / "backup.sqlite3"
        self.store.backup(destination)
        self.assertEqual(RehabStore(destination).patient(self.patient["id"])["name"], self.patient["name"])


class ApiTests(unittest.TestCase):
    def test_business_available_with_failed_model(self):
        from fastapi.testclient import TestClient
        from rehab_service.api import rtc as server
        from rehab_service.app import app
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/db.sqlite3"}), \
                patch.object(server, "engine", None), patch.object(server, "PoseEngine", side_effect=RuntimeError("missing model")):
            with TestClient(app) as client:
                token = client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
                client.headers["Authorization"] = "Bearer " + token
                patient = client.post("/api/v1/patients", json={"name": "测试", "affected_side": "left", "forearm_cm": 25})
                self.assertEqual(patient.status_code, 201)
                identity = patient.json()["id"]
                self.assertEqual(len(client.get("/api/v1/actions").json()), 6)
                self.assertEqual(client.get("/api/v1/patients/" + identity + "/plan").json()["items"], [])
                self.assertEqual(client.get("/api/v1/patients/missing").status_code, 404)
                self.assertEqual(client.post("/rtc/offer", json={}).status_code, 503)
                self.assertEqual(client.get("/api/v1/patients/" + identity + "/overview").status_code, 200)


if __name__ == "__main__":
    unittest.main()
