"""Patient credentials, staff ownership and independent email recovery."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehab_service.services.auth import AuthService
from rehab_service.storage.database import BusinessError, RehabStore


class PatientPasswordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "db.sqlite3"
        self.store = RehabStore(self.path)
        self.auth = AuthService(self.store)
        self.patient = self.store.save_patient({"name": "密码测试", "affected_side": "left", "forearm_cm": 25, "email": "one@example.com"})
        self.id = self.patient["id"]
        self.payload = {"username": "patient.one", "password": "Test-pass-1"}
        self.auth.save_patient_account(self.id, self.payload)

    def tearDown(self):
        self.temp.cleanup()

    def reset_code(self, side_effect=None):
        with patch("rehab_service.services.auth.send_mail", side_effect=side_effect) as mail:
            response = self.auth.password_reset_code({"username": "patient.one", "email": "one@example.com"}, "test")
            self.assertNotIn("code", response)
            self.assertEqual(mail.call_args.args[2], "reset")
            return mail.call_args.args[1]

    def reset(self, code, **values):
        return self.auth.password_reset(dict(username="patient.one", email="one@example.com", code=code, password="New-pass-2", **values), "test")

    def test_credential_login_hash_update_and_revoke(self):
        login = self.auth.patient_login(dict(self.payload, username="PATIENT.ONE"), "test")
        self.assertEqual(login["patient_id"], self.id)
        with self.store.db() as db:
            saved = db.execute("SELECT password FROM accounts WHERE patient_id=?", (self.id,)).fetchone()[0]
            self.assertNotEqual(saved, self.payload["password"])
        account = self.auth.patient_account(self.id)
        self.assertNotIn("password", account)
        self.auth.save_patient_account(self.id, {"username": "patient.two"})
        with self.assertRaises(BusinessError): self.auth.user("Bearer " + login["token"])
        self.auth.patient_login(dict(self.payload, username="patient.two"), "test")
        self.auth.save_patient_account(self.id, {"username": "patient.two", "password": "Reset-by-admin"})
        with self.assertRaises(BusinessError): self.auth.patient_login(dict(self.payload, username="patient.two"), "test")
        self.auth.patient_login({"username": "patient.two", "password": "Reset-by-admin"}, "test")

    def test_existing_otp_account_id_and_patient_history_survive(self):
        with self.store.db() as db:
            account_id = db.execute("SELECT id FROM accounts WHERE patient_id=?", (self.id,)).fetchone()[0]
            db.execute("UPDATE accounts SET username=NULL,password=NULL WHERE id=?", (account_id,))
        self.auth.save_patient_account(self.id, self.payload)
        with patch("rehab_service.services.auth.send_mail") as mail:
            self.auth.email_code({"email": "one@example.com"}, "test")
            login = self.auth.email_login({"email": "one@example.com", "code": mail.call_args.args[1]}, "test")
        self.assertEqual(self.auth.user("Bearer " + login["token"])["id"], account_id)
        self.assertEqual(self.store.patient(self.id)["name"], "密码测试")
        self.auth.patient_login(self.payload, "test")

    def test_unique_accounts_validation_and_no_staff_as_patient(self):
        second = self.store.save_patient({"name": "患者乙", "affected_side": "right", "forearm_cm": 25})
        for payload in ({"username": "patient.one", "password": "long-password"},
                        {"username": "admin", "password": "long-password"},
                        {"username": "new.patient"}, {"username": "a/b", "password": "long-password"},
                        {"username": "new.patient", "password": "short"}):
            with self.assertRaises(BusinessError): self.auth.save_patient_account(second["id"], payload)
        with self.assertRaises(BusinessError): self.auth.patient_login({"username": "admin", "password": "123"}, "test")
        self.auth.staff_login({"username": "admin", "password": "123"}, "test")

    def test_reset_is_single_use_revokes_tokens_and_login_code(self):
        login = self.auth.patient_login(self.payload, "test")
        with patch("rehab_service.services.auth.send_mail") as mail:
            self.auth.email_code({"email": "one@example.com"}, "test")
            login_code = mail.call_args.args[1]
        code = self.reset_code()
        with self.store.db() as db:
            self.assertNotIn(code, db.execute("SELECT digest FROM password_reset_codes").fetchone()[0])
        self.reset(code)
        for action in (lambda: self.reset(code), lambda: self.auth.user("Bearer " + login["token"]),
                       lambda: self.auth.patient_login(self.payload, "test"),
                       lambda: self.auth.email_login({"email": "one@example.com", "code": login_code}, "test")):
            with self.assertRaises(BusinessError): action()
        self.auth.patient_login(dict(self.payload, password="New-pass-2"), "test")

    def test_reset_and_login_challenges_cannot_be_interchanged(self):
        with patch("rehab_service.services.auth.send_mail") as mail, patch("rehab_service.services.auth.secrets.randbelow", return_value=123456):
            self.auth.email_code({"email": "one@example.com"}, "test")
        code = self.reset_code()
        wrong = "000000" if code != "000000" else "111111"
        with self.assertRaises(BusinessError): self.reset(wrong)
        with self.store.db() as db: self.assertEqual(db.execute("SELECT attempts FROM password_reset_codes").fetchone()[0], 1)
        # Login-only challenge never authorizes a password reset after reset challenge is deleted.
        with self.store.db() as db: db.execute("DELETE FROM password_reset_codes")
        with self.assertRaises(BusinessError): self.reset("123456")

    def test_attempts_expiration_resend_and_failed_mail(self):
        code = self.reset_code(); wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            with self.assertRaises(BusinessError): self.reset(wrong)
        with self.assertRaises(BusinessError): self.reset(code)
        with self.assertRaises(BusinessError) as error: self.reset_code()
        self.assertEqual(error.exception.status, 429)
        with self.store.db() as db:
            db.execute("DELETE FROM auth_limits")
            db.execute("UPDATE password_reset_codes SET attempts=0,expires=?", (time.time() - 1,))
        with self.assertRaises(BusinessError): self.reset(code)
        with patch("rehab_service.services.auth.secrets.randbelow", return_value=654321): new_code = self.reset_code()
        with self.store.db() as db: db.execute("DELETE FROM auth_limits")
        with self.assertRaises(BusinessError): self.reset_code(BusinessError("发送失败", 503))
        self.reset(new_code)

    def test_admin_reset_and_email_rebind_invalidate_recovery(self):
        code = self.reset_code()
        self.auth.save_patient_account(self.id, dict(self.payload, password="Admin-pass-2"))
        with self.assertRaises(BusinessError): self.reset(code)
        with self.store.db() as db: db.execute("DELETE FROM auth_limits")
        code = self.reset_code()
        self.store.save_patient({"email": "new@example.com"}, self.id)
        with self.assertRaises(BusinessError): self.reset(code)
        with self.store.db() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM password_reset_codes").fetchone()[0], 0)

    def test_account_changed_during_smtp_cannot_publish_code(self):
        def change(*args): self.auth.save_patient_account(self.id, dict(self.payload, password="Admin-pass-2"))
        with self.assertRaises(BusinessError): self.reset_code(change)
        with self.store.db() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM password_reset_codes").fetchone()[0], 0)

    def test_login_throttle_and_archived_patient(self):
        for _ in range(10):
            with self.assertRaises(BusinessError): self.auth.patient_login(dict(self.payload, password="wrong"), "test")
        with self.assertRaises(BusinessError) as error: self.auth.patient_login(self.payload, "test")
        self.assertEqual(error.exception.status, 429)
        self.store.save_patient({"archived": True}, self.id)
        with self.store.db() as db: db.execute("DELETE FROM auth_limits")
        with self.assertRaises(BusinessError): self.auth.patient_login(self.payload, "test")
        with self.assertRaises(BusinessError): self.reset_code()

    def test_v2_upgrade_keeps_credentials_and_only_runs_once(self):
        with self.store.db() as db:
            db.execute("DROP TABLE password_reset_codes"); db.execute("PRAGMA user_version=2")
        upgraded = RehabStore(self.path)
        self.assertEqual(AuthService(upgraded).patient_login(self.payload, "test")["patient_id"], self.id)
        backups = list(Path(self.temp.name).glob("*.before-v3-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        with sqlite3.connect(backups[0]) as db: self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
        RehabStore(self.path)
        self.assertEqual(len(list(Path(self.temp.name).glob("*.before-v3-*.sqlite3"))), 1)


class PatientAccountApiTests(unittest.TestCase):
    def test_only_staff_manages_credentials_and_patient_stays_scoped(self):
        from fastapi.testclient import TestClient
        from rehab_service.api import rtc as server
        from rehab_service.app import app
        from test_rtc import FakeEngine
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"REHAB_DB": directory + "/qa.sqlite3"}), patch.object(server, "engine", FakeEngine()), TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + client.post("/api/v1/auth/staff-login", json={"username": "admin", "password": "123"}).json()["token"]
            patient = client.post("/api/v1/patients", json={"name": "账号患者", "affected_side": "left", "forearm_cm": 25, "email": "qa@example.com"}).json()
            other = client.post("/api/v1/patients", json={"name": "其他患者", "affected_side": "right", "forearm_cm": 25}).json()
            path = "/api/v1/patients/" + patient["id"] + "/account"
            credentials = {"username": "qa.patient", "password": "Password-1"}
            self.assertEqual(client.put(path, json=credentials).status_code, 200)
            self.assertNotIn("password", client.get(path).json())
            login = client.post("/api/v1/auth/patient-login", json=credentials).json()
            client.headers["Authorization"] = "Bearer " + login["token"]
            self.assertEqual(client.get(path).status_code, 403)
            self.assertEqual(client.put(path, json=credentials).status_code, 403)
            self.assertEqual(client.get("/api/v1/patients/" + patient["id"]).status_code, 200)
            self.assertEqual(client.get("/api/v1/patients/" + other["id"]).status_code, 403)
            with patch("rehab_service.services.auth.send_mail") as mail:
                self.assertEqual(client.post("/api/v1/auth/password-reset-code", json={"username": "qa.patient", "email": "qa@example.com"}).status_code, 200)
                code = mail.call_args.args[1]
            self.assertEqual(client.post("/api/v1/auth/password-reset", json={"username": "qa.patient", "email": "qa@example.com", "code": code, "password": "Password-2"}).status_code, 200)
            self.assertEqual(client.get("/api/v1/auth/me").status_code, 401)
            self.assertEqual(client.post("/api/v1/auth/patient-login", json=dict(credentials, password="Password-2")).status_code, 200)


if __name__ == "__main__":
    unittest.main()
