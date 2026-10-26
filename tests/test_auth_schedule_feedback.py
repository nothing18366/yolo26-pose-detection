"""Authentication, schedule, feedback, versioning and migration contracts."""
import os
from datetime import date, timedelta
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.storage.database import RehabStore, BusinessError, now
from rehab_service.services.auth import AuthService, send_mail
from rehab_service.services.teaching import TeachingService


class NextStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = RehabStore(Path(self.temp.name) / "db.sqlite3")
        self.auth = AuthService(self.store)
        self.patient = self.store.save_patient({"name": "患者甲", "affected_side": "left", "forearm_cm": 25, "email": "A@example.com"})
        self.id = self.patient["id"]

    def tearDown(self):
        self.temp.cleanup()

    def code(self):
        with patch("rehab_service.services.auth.send_mail") as mail:
            self.auth.email_code({"email": "a@example.com"}, "test-ip")
            return mail.call_args.args[1]

    def login(self):
        return self.auth.email_login({"email": "a@example.com", "code": self.code()}, "test-ip")

    def run_one(self, mode="plan", count=0, reason="manual", pre=None):
        run = self.store.create_run({"patient_id": self.id, "mode": mode, "action": "cup", "target_reps": 10}, "patient")
        self.store.feedback(run["id"], "pre", pre or {"state": "skipped"}, "tester")
        session = self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.claim_session(session["id"])
        self.store.finish_session(session["id"], {"count": count, "passed": count // 2, "target": 10, "total": 2}, 30, now(), reason)
        return self.store.run(run["id"])

    def plan(self, **values):
        return self.store.save_plan(self.id, dict(items=[{"action": "cup", "target_reps": 10}], weekdays=[1, 3, 5], **values))

    def test_code_hash_single_use_and_token_revocation(self):
        code = self.code()
        with self.store.db() as db:
            row = db.execute("SELECT * FROM email_codes").fetchone()
            self.assertNotIn(code, row["digest"])
        login = self.auth.email_login({"email": "a@example.com", "code": code}, "test-ip")
        user = self.auth.user("Bearer " + login["token"])
        self.assertEqual(user["patient_id"], self.id)
        self.assertAlmostEqual(login["expires"] - time.time(), 86400, delta=2)
        with self.assertRaises(BusinessError): self.auth.email_login({"email": "a@example.com", "code": code}, "test-ip")
        self.auth.logout(user)
        with self.assertRaises(BusinessError): self.auth.user("Bearer " + login["token"])

    def test_wrong_attempts_expiry_throttle_and_failure(self):
        code = self.code()
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            with self.assertRaises(BusinessError): self.auth.email_login({"email": "a@example.com", "code": wrong}, "test-ip")
        with self.assertRaises(BusinessError): self.auth.email_login({"email": "a@example.com", "code": code}, "test-ip")
        with self.assertRaises(BusinessError) as e: self.auth.email_code({"email": "a@example.com"}, "test-ip")
        self.assertEqual(e.exception.status, 429)
        with self.store.db() as db:
            db.execute("DELETE FROM auth_limits")
            db.execute("UPDATE email_codes SET expires=?,attempts=0", (time.time() - 1,))
        with self.assertRaises(BusinessError): self.auth.email_login({"email": "a@example.com", "code": code}, "test-ip")
        with patch("rehab_service.services.auth.send_mail", side_effect=BusinessError("SMTP失败", 503)):
            with self.assertRaises(BusinessError): self.auth.email_code({"email": "a@example.com"}, "test-ip")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(BusinessError) as e: send_mail("a@example.com", "123456")
        self.assertEqual(e.exception.status, 503)

    def test_smtp_ssl_and_starttls_use_real_transport_contract(self):
        for security, transport in (("ssl", "SMTP_SSL"), ("starttls", "SMTP")):
            with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com", "SMTP_FROM": "service@example.com", "SMTP_PASSWORD": "test-password", "SMTP_SECURITY": security}), patch("rehab_service.services.auth.smtplib." + transport) as smtp:
                send_mail("a@example.com", "123456")
                connection = smtp.return_value.__enter__.return_value
                connection.login.assert_called_once()
                connection.send_message.assert_called_once()
                self.assertEqual(connection.starttls.call_count, int(security == "starttls"))
        with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com", "SMTP_FROM": "service@example.com", "SMTP_PASSWORD": "test", "SMTP_SECURITY": "starttls"}), patch("rehab_service.services.auth.smtplib.SMTP", side_effect=OSError("offline")):
            with self.assertRaises(BusinessError) as error: send_mail("a@example.com", "123456")
            self.assertEqual(error.exception.status, 503)

    def test_email_rebind_during_send_does_not_publish_challenge(self):
        def rebind(email, code): self.store.save_patient({"email": "new@example.com"}, self.id)
        with patch("rehab_service.services.auth.send_mail", side_effect=rebind):
            with self.assertRaises(BusinessError): self.auth.email_code({"email": "a@example.com"}, "test-ip")
        with self.store.db() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM email_codes").fetchone()[0], 0)

    def test_rebinding_unique_email_revokes_login(self):
        login = self.login()
        with self.assertRaises(BusinessError): self.store.save_patient({"name": "患者乙", "affected_side": "right", "forearm_cm": 26, "email": "a@example.com"})
        self.store.save_patient({"email": "new@example.com"}, self.id)
        self.assertEqual(self.store.patient(self.id)["id"], self.id)
        with self.assertRaises(BusinessError): self.auth.user("Bearer " + login["token"])

    def test_staff_password_and_expired_token(self):
        login = self.auth.staff_login({"username": "admin", "password": "123"}, "test-ip")
        self.assertEqual(login["role"], "staff")
        self.assertAlmostEqual(login["expires"] - time.time(), 8 * 3600, delta=2)
        with self.store.db() as db:
            self.assertNotEqual(db.execute("SELECT password FROM accounts WHERE username='admin'").fetchone()[0], "123")
            db.execute("UPDATE login_sessions SET expires=0")
        with self.assertRaises(BusinessError): self.auth.user("Bearer " + login["token"])

    def test_schedule_completion_denominator_and_pending_history(self):
        monday = date(2026, 10, 19)
        with patch.object(RehabStore, "today", return_value=monday): self.plan()
        with patch.object(RehabStore, "today", return_value=monday + timedelta(days=2)):
            self.run_one(count=0)  # All actions normal, even below repetition target, completes this scheduled day.
            self.run_one(count=4)
            self.run_one("single", 4)
            self.assertEqual(self.store.calendar(self.id)["completed_days"], 1)
            self.assertEqual(self.store.calendar(self.id)["due_days"], 2)
            self.assertEqual(self.store.calendar(self.id)["scheduled_days"], 3)
            self.assertEqual(self.store.calendar(self.id)["adherence"], 50)
            self.assertEqual(self.store.calendar(self.id)["target_reached_days"], 0)
            self.run_one(count=10)
            self.assertEqual(self.store.calendar(self.id)["completed_days"], 1)
            self.assertEqual(self.store.calendar(self.id)["target_reached_days"], 1)
            with self.assertRaises(BusinessError): self.plan(effective_from="2026-10-21")
            current = self.plan(effective_from="2026-10-22")
            self.assertEqual(current["revision"], 1)
            self.assertEqual(current["pending"]["revision"], 2)
            self.plan(effective_from="2026-10-23")
            with self.store.db() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM plan_versions").fetchone()[0], 2)
        with patch.object(RehabStore, "today", return_value=date(2026, 10, 23)):
            self.assertNotEqual(self.store.plan(self.id)["revision"], 1)
            self.assertEqual(self.store.calendar(self.id)["completed_days"], 1)

    def test_first_future_plan_no_premature_schedule_and_legacy_no_backfill(self):
        with patch.object(RehabStore, "today", return_value=date(2026, 10, 19)):
            self.plan(effective_from="2026-10-23")
            self.assertEqual(self.store.plan(self.id)["items"], [])
            self.assertEqual(self.store.calendar(self.id)["due_days"], 0)
            with self.store.db() as db:
                db.execute("DELETE FROM plan_versions")
                db.execute("INSERT INTO plans VALUES (?,?,?,?)", (self.id, '[{"action":"cup","target_reps":10}]', 1, now()))
            self.assertEqual(self.store.calendar(self.id)["scheduled_days"], 0)
            self.assertFalse(self.store.plan(self.id)["scheduled"])
            self.run_one()

    def test_feedback_null_skip_immutability_and_discomfort(self):
        self.plan()
        run = self.store.create_run({"patient_id": self.id, "mode": "plan"})
        with self.assertRaises(BusinessError): self.store.create_session({"run_id": run["id"], "item_index": 0})
        self.store.feedback(run["id"], "pre", {"state": "submitted", "pain": 0}, "patient")
        f = self.store.run(run["id"])["feedback"]["pre"]
        self.assertEqual(f["data"]["pain"], 0)
        self.assertIsNone(f["data"]["fatigue"])
        self.store.feedback(run["id"], "pre", {"state": "skipped"}, "patient")
        self.assertEqual(self.store.run(run["id"])["feedback"]["pre"]["data"], {})
        s = self.store.create_session({"run_id": run["id"], "item_index": 0}); self.store.claim_session(s["id"])
        with self.assertRaises(BusinessError): self.store.feedback(run["id"], "pre", {"state": "skipped"}, "patient")
        self.store.finish_session(s["id"], {"count": 4, "target": 10}, 20, now(), "user_discomfort")
        self.assertEqual(self.store.run(run["id"])["status"], "unfinished")
        for pain in (3, 2): self.store.feedback(run["id"], "post", {"state": "submitted", "pain": pain, "difficulty": 4}, "patient")
        with self.store.db() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM feedback WHERE phase='post'").fetchone()[0], 1)
        self.assertEqual(self.store.run(run["id"])["sessions"][0]["summary"]["count"], 4)
        self.assertEqual(self.store.calendar(self.id)["completed_days"], 0)

    def test_cross_midnight_stays_on_start_date(self):
        with patch.object(RehabStore, "today", return_value=date(2026, 10, 21)):
            self.plan(); run = self.store.create_run({"patient_id": self.id, "mode": "plan"})
            self.store.feedback(run["id"], "pre", {"state": "skipped"}, "patient")
            session = self.store.create_session({"run_id": run["id"], "item_index": 0}); self.store.claim_session(session["id"])
        with patch.object(RehabStore, "today", return_value=date(2026, 10, 22)):
            self.store.finish_session(session["id"], {"count": 0, "target": 10}, 120, now())
            self.assertEqual(self.store.run(run["id"])["training_date"], "2026-10-21")
            self.assertTrue(self.store.calendar(self.id)["days"][2]["completed"])

    def test_trends_group_conditions_partial_zero_and_legacy(self):
        self.plan(); self.run_one(count=4); self.run_one(count=100, reason="disconnected"); self.run_one(count=0)
        self.store.save_patient({"forearm_cm": 27}, self.id); legacy = self.run_one(count=6)
        response = self.store.trends(self.id, "cup", "2026-01-01", "2026-12-31")
        self.assertEqual(len(response["groups"]), 2)
        self.assertEqual(response["groups"][0]["count"], 4)
        self.assertEqual(response["groups"][0]["rate"], 50)
        self.assertIsNone(response["groups"][0]["points"][2]["rate"])
        with self.store.db() as db: db.execute("UPDATE runs SET conditions=NULL WHERE id=?", (legacy["id"],))
        self.assertEqual(len(self.store.trends(self.id, "cup", "2026-01-01", "2026-12-31")["legacy"]), 1)
        with self.assertRaises(BusinessError): self.store.create_run({"patient_id": self.id, "mode": "single", "action": "press", "target_reps": 8}, "patient")
        with self.assertRaises(BusinessError): self.store.create_run({"patient_id": self.id, "mode": "single", "action": "cup", "target_reps": 20}, "patient")

    def test_teaching_new_asset_invalidates_review(self):
        root = Path(self.temp.name) / "media"; root.mkdir(); (root / "cup.srt").write_text("version1")
        teaching = TeachingService(self.store, root); item = next(i for i in teaching.catalog() if i["key"] == "cup")
        self.assertEqual(item["review"]["state"], "pending")
        teaching.review("cup", {"content_version": item["content_version"], "state": "approved", "reviewer": "工作人员", "note": "核对"}, "admin")
        self.assertEqual(next(i for i in teaching.catalog() if i["key"] == "cup")["review"]["state"], "approved")
        (root / "cup.srt").write_text("version2")
        changed = TeachingService(self.store, root)
        self.assertEqual(next(i for i in changed.catalog() if i["key"] == "cup")["review"]["state"], "pending")
        with self.assertRaises(BusinessError): changed.review("cup", {"content_version": item["content_version"]}, "admin")


class PermissionTests(unittest.TestCase):
    def test_two_patients_cannot_cross_data_or_mutate_plans_or_anonymous_rtc(self):
        from fastapi.testclient import TestClient
        from rehab_service.api import rtc as server
        from rehab_service.app import app
        from test_rtc import FakeEngine
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/db.sqlite3"}), patch.object(server, "engine", FakeEngine()), TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
            patients = []
            for n in range(2):
                p = client.post("/api/v1/patients", json={"name": "患者" + str(n), "affected_side": "left", "forearm_cm": 25, "email": str(n) + "@example.com"}).json(); patients.append(p)
                client.put("/api/v1/patients/" + p["id"] + "/plan", json={"items": [{"action": "cup", "target_reps": 10}], "weekdays": [1,3,5]})
            other_run = client.post("/api/v1/training-runs", json={"patient_id": patients[1]["id"], "mode": "plan"}).json()
            client.put("/api/v1/training-runs/" + other_run["id"] + "/feedback/pre", json={"state": "skipped"})
            other_session = client.post("/api/v1/sessions", json={"run_id": other_run["id"], "item_index": 0}).json()
            with patch("rehab_service.services.auth.send_mail") as mail:
                response = client.post("/api/v1/auth/email-code", json={"email": "0@example.com"})
                self.assertEqual(response.status_code, 200); self.assertNotIn("code", response.json())
                code = mail.call_args.args[1]
            login = client.post("/api/v1/auth/email-login", json={"email": "0@example.com", "code": code}).json()
            client.headers["Authorization"] = "Bearer " + login["token"]
            self.assertEqual(client.get("/api/v1/patients/" + patients[0]["id"]).status_code, 200)
            for path in ("/patients/" + patients[1]["id"], "/training-runs/" + other_run["id"], "/sessions/" + other_session["id"], "/patients"):
                self.assertEqual(client.get("/api/v1" + path).status_code, 403, path)
            self.assertEqual(client.put("/api/v1/patients/" + patients[0]["id"] + "/plan", json={}).status_code, 403)
            self.assertEqual(client.patch("/api/v1/patients/" + patients[0]["id"], json={"name": "禁止修改"}).status_code, 403)
            self.assertEqual(client.post("/api/v1/patients", json={}).status_code, 403)
            self.assertEqual(client.patch("/api/v1/actions/cup/review", json={}).status_code, 403)
            self.assertEqual(client.get("/api/v1/patients/" + patients[1]["id"] + "/calendar").status_code, 403)
            self.assertEqual(client.get("/api/v1/patients/" + patients[1]["id"] + "/trends?action=cup&start=2026-01-01&end=2026-12-31").status_code, 403)
            self.assertEqual(client.post("/api/v1/sessions/" + other_session["id"] + "/interrupt", json={}).status_code, 403)
            self.assertEqual(client.post("/api/v1/training-runs/" + other_run["id"] + "/finish").status_code, 403)
            self.assertEqual(client.put("/api/v1/training-runs/" + other_run["id"] + "/feedback/pre", json={"state": "skipped"}).status_code, 403)
            self.assertEqual(client.post("/rtc/offer", json={"type": "offer", "sdp": "bad", "session_id": other_session["id"]}).status_code, 403)
            self.assertEqual(client.post("/rtc/offer", json={"type": "offer", "sdp": "bad"}).status_code, 400)
            client.headers.pop("Authorization")
            self.assertEqual(client.get("/api/v1/training-runs/" + other_run["id"]).status_code, 401)
            self.assertEqual(client.post("/rtc/offer", json={}).status_code, 401)
            self.assertEqual(client.get("/media/cup.srt").status_code, 200)


class MigrationTests(unittest.TestCase):
    def test_v1_backup_preserves_patient_and_run_ids_and_only_migrates_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.sqlite3"
            with sqlite3.connect(path) as db:
                db.executescript("""CREATE TABLE patients (id TEXT PRIMARY KEY,name TEXT NOT NULL,affected_side TEXT NOT NULL,
                    forearm_cm REAL NOT NULL,archived INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                    CREATE TABLE runs (id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,patient TEXT NOT NULL,items TEXT NOT NULL,mode TEXT NOT NULL,
                    plan_revision INTEGER,status TEXT NOT NULL,created_at TEXT NOT NULL,ended_at TEXT); PRAGMA user_version=1;""")
                db.execute("INSERT INTO patients VALUES ('original-patient','保留患者','left',25,0,?,?)", (now(), now()))
                db.execute("INSERT INTO runs VALUES ('original-run','original-patient','{}','[]','plan',1,'completed',?,?)", (now(), now()))
            migrated = RehabStore(path)
            self.assertEqual(migrated.patient("original-patient")["name"], "保留患者")
            self.assertIsNone(migrated.run("original-run")["conditions"])
            backups = list(Path(folder).glob("*.before-v2-*.sqlite3")); self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0]) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT id FROM runs").fetchone()[0], "original-run")
            RehabStore(path)
            self.assertEqual(len(list(Path(folder).glob("*.before-v2-*.sqlite3"))), 1)


if __name__ == "__main__": unittest.main()
