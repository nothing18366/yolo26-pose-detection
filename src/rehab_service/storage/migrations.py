"""Append-only SQLite upgrades; back up existing databases before changing schema."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3


def upgrade(path, existed):
    with sqlite3.connect(path) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version >= 3:
            return
        if existed:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            backup = Path(str(path) + (".before-v2-" if version < 2 else ".before-v3-") + stamp + ".sqlite3")
            with sqlite3.connect(backup) as target:
                db.backup(target)
        if version < 2:
            db.executescript("""
            BEGIN IMMEDIATE;
            ALTER TABLE patients ADD COLUMN email TEXT;
            CREATE UNIQUE INDEX patients_email ON patients(email) WHERE email IS NOT NULL;
            ALTER TABLE runs ADD COLUMN training_date TEXT;
            ALTER TABLE runs ADD COLUMN schedule_revision INTEGER;
            ALTER TABLE runs ADD COLUMN conditions TEXT;
            ALTER TABLE runs ADD COLUMN teaching TEXT;
            CREATE TABLE accounts (
                id TEXT PRIMARY KEY, role TEXT NOT NULL, username TEXT UNIQUE,
                patient_id TEXT UNIQUE REFERENCES patients(id), password TEXT, created_at TEXT NOT NULL);
            CREATE TABLE login_sessions (
                token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
                expires REAL NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE email_codes (
                email TEXT PRIMARY KEY, digest TEXT NOT NULL, expires REAL NOT NULL,
                attempts INTEGER NOT NULL, sent_at REAL NOT NULL);
            CREATE TABLE auth_limits (bucket TEXT NOT NULL, at REAL NOT NULL);
            CREATE INDEX auth_limits_bucket ON auth_limits(bucket, at);
            CREATE TABLE plan_versions (
                patient_id TEXT NOT NULL REFERENCES patients(id), revision INTEGER NOT NULL,
                effective_from TEXT NOT NULL, weekdays TEXT NOT NULL, items TEXT NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(patient_id, revision));
            CREATE TABLE feedback (
                run_id TEXT NOT NULL REFERENCES runs(id), phase TEXT NOT NULL,
                state TEXT NOT NULL, data TEXT NOT NULL, author TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY(run_id, phase));
            CREATE TABLE teaching_reviews (
                action TEXT NOT NULL, version TEXT NOT NULL, state TEXT NOT NULL,
                reviewer TEXT, reviewed_at TEXT, note TEXT NOT NULL, author TEXT NOT NULL,
                PRIMARY KEY(action, version));
            PRAGMA user_version=2;
            COMMIT;
            """)
        db.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE password_reset_codes (
                email TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
                digest TEXT NOT NULL, expires REAL NOT NULL, attempts INTEGER NOT NULL,
                sent_at REAL NOT NULL);
            PRAGMA user_version=3;
            COMMIT;
        """)
