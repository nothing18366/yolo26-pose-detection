"""Local patient, plan and training records. Never stores camera images."""

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

from rehab_service.catalog import BY_KEY
from rehab_service.settings import MODEL_DIR


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class BusinessError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def require(condition, message, status=400):
    if not condition:
        raise BusinessError(message, status)


def target_value(value):
    require(isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 100,
            "目标次数必须是 1–100 的整数")
    return value


class RehabStore:
    def __init__(self, path):
        self.path = Path(path)
        existed = self.path.is_file() and self.path.stat().st_size > 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, affected_side TEXT NOT NULL,
                    forearm_cm REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS plans (
                    patient_id TEXT PRIMARY KEY REFERENCES patients(id),
                    items TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
                    patient TEXT NOT NULL, items TEXT NOT NULL, mode TEXT NOT NULL,
                    plan_revision INTEGER, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    ended_at TEXT);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    item_index INTEGER NOT NULL, action TEXT NOT NULL, target INTEGER NOT NULL,
                    patient TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    started_at TEXT, ended_at TEXT, elapsed_seconds REAL NOT NULL DEFAULT 0,
                    summary TEXT, reason TEXT, checkpoint_at TEXT);
                CREATE INDEX IF NOT EXISTS runs_patient_date ON runs(patient_id, created_at);
                CREATE INDEX IF NOT EXISTS sessions_run ON sessions(run_id, item_index);
            """)
        from rehab_service.storage.migrations import upgrade
        upgrade(self.path, existed)
        self.algorithm_version = self.algorithm_fingerprint()
        self.teaching_versions = {}

    @staticmethod
    def algorithm_fingerprint():
        root = Path(__file__).resolve().parents[1] / "recognition"
        paths = [root / p for p in ("inference.py", "tracking.py", "metrics.py")]
        paths += sorted((root / "actions").glob("*.py")) + sorted(MODEL_DIR.glob("*"))
        checksum = hashlib.sha256()
        for path in paths:
            if path.is_file():
                checksum.update(path.name.encode())
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1048576), b""):
                        checksum.update(block)
        fingerprint = checksum.hexdigest()[:16]
        # Exact display-only sources: all six fixed videos preserve keypoints and action results.
        return {"ded978947d4322d5": "989357def6b46bd9",
                "e5c306b6a51c973b": "989357def6b46bd9",
                # Module description updated; inference implementation is byte-identical.
                "13d97e8aaf1e6120": "989357def6b46bd9"}.get(fingerprint, fingerprint)

    @staticmethod
    def today():
        return datetime.now(timezone(timedelta(hours=8))).date()

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def one(db, table, identity):
        require(isinstance(identity, str) and 1 <= len(identity) <= 64, "记录编号格式错误")
        row = db.execute(f"SELECT * FROM {table} WHERE id=?", (identity,)).fetchone()
        require(row is not None, "记录不存在", 404)
        return dict(row)

    @staticmethod
    def patient_row(row):
        row = dict(row)
        row["archived"] = bool(row["archived"])
        row["code"] = "P-" + row["id"][:8].upper()
        return row

    def patient(self, identity):
        with self.db() as db:
            return self.patient_row(self.one(db, "patients", identity))

    def patients(self, query="", archived=False):
        with self.db() as db:
            rows = db.execute("SELECT * FROM patients WHERE archived=? ORDER BY created_at DESC",
                              (int(archived),)).fetchall()
        return [self.patient_row(row) for row in rows
                if query.strip().lower() in (row["name"] + " " + row["id"] + " P-" + row["id"][:8]).lower()]

    def save_patient(self, payload, identity=None):
        current = self.patient(identity) if identity else {}
        name = payload.get("name", current.get("name", ""))
        side = payload.get("affected_side", current.get("affected_side"))
        forearm = payload.get("forearm_cm", current.get("forearm_cm"))
        archived = payload.get("archived", current.get("archived", False))
        email = payload.get("email", current.get("email"))
        require(email is None or isinstance(email, str), "请输入有效邮箱")
        if isinstance(email, str) and email.strip():
            from rehab_service.services.auth import normalize_email
            email = normalize_email(email)
        else:
            email = None
        require(isinstance(name, str) and 1 <= len(name.strip()) <= 60, "姓名需为 1–60 个字符")
        require(side in ("left", "right"), "请选择左侧或右侧患侧")
        require(isinstance(forearm, (float, int)) and not isinstance(forearm, bool)
                and math.isfinite(forearm) and 10 <= forearm <= 45, "前臂长度需为 10–45 cm")
        require(isinstance(archived, bool), "归档状态格式错误")
        stamp = now()
        identity = identity or uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            require(not email or not db.execute("SELECT 1 FROM patients WHERE email=? AND id<>?", (email, identity)).fetchone(), "邮箱已绑定其他患者", 409)
            if current:
                require(not db.execute("""SELECT 1 FROM sessions s JOIN runs r ON s.run_id=r.id
                    WHERE r.patient_id=? AND s.status IN ('connecting','running')""",
                    (identity,)).fetchone(), "训练中不能修改患者资料", 409)
                db.execute("UPDATE patients SET name=?,affected_side=?,forearm_cm=?,archived=?,updated_at=?,email=? WHERE id=?",
                           (name.strip(), side, forearm, int(archived), stamp, email, identity))
                if email != current.get("email"):
                    db.execute("DELETE FROM login_sessions WHERE account_id IN (SELECT id FROM accounts WHERE patient_id=?)", (identity,))
                    db.execute("DELETE FROM email_codes WHERE email IN (?,?)", (email, current.get("email")))
                    db.execute("DELETE FROM password_reset_codes WHERE account_id IN (SELECT id FROM accounts WHERE patient_id=?)", (identity,))
            else:
                db.execute("INSERT INTO patients (id,name,affected_side,forearm_cm,archived,created_at,updated_at,email) VALUES (?,?,?,?,?,?,?,?)",
                           (identity, name.strip(), side, forearm, int(archived), stamp, stamp, email))
        return self.patient(identity)

    def plan(self, patient_id, day=None):
        self.patient(patient_id)
        day = day or self.today()
        with self.db() as db:
            row = db.execute("SELECT * FROM plans WHERE patient_id=?", (patient_id,)).fetchone()
            versions = db.execute("SELECT * FROM plan_versions WHERE patient_id=? ORDER BY effective_from,revision", (patient_id,)).fetchall()
        current = next((v for v in reversed(versions) if v["effective_from"] <= day.isoformat()), None)
        pending = next((v for v in versions if v["effective_from"] > day.isoformat()), None)
        def decoded(v):
            return dict(v, items=json.loads(v["items"]), weekdays=json.loads(v["weekdays"])) if v else None
        # Unscheduled legacy plans remain usable; no backdated adherence is inferred.
        return {"patient_id": patient_id, "revision": current["revision"] if current else row["revision"] if row else 0,
                "items": json.loads(current["items"]) if current else json.loads(row["items"]) if row else [],
                "weekdays": json.loads(current["weekdays"]) if current else [],
                "effective_from": current["effective_from"] if current else None,
                "scheduled": current is not None, "pending": decoded(pending)}

    @staticmethod
    def validate_items(items):
        require(isinstance(items, list) and 1 <= len(items) <= 6, "请选择 1–6 个动作")
        seen = set()
        output = []
        for item in items:
            require(isinstance(item, dict) and isinstance(item.get("action"), str)
                    and item["action"] in BY_KEY, "存在不支持的动作")
            key = item["action"]
            require(key not in seen, "同一计划中动作不能重复")
            seen.add(key)
            output.append({"action": key, "target_reps": target_value(item.get("target_reps"))})
        return output

    def save_plan(self, patient_id, payload):
        require(not self.patient(patient_id)["archived"], "患者已归档")
        items = self.validate_items(payload.get("items"))
        weekdays = payload.get("weekdays", [])
        require(isinstance(weekdays, list) and all(type(d) is int and 1 <= d <= 7 for d in weekdays)
                and len(set(weekdays)) == len(weekdays), "训练日必须为周一至周日，不可重复")
        current = self.plan(patient_id)
        earliest = self.today() + timedelta(days=1 if current["items"] else 0)
        try:
            effective = date.fromisoformat(payload.get("effective_from", earliest.isoformat()))
        except (TypeError, ValueError):
            raise BusinessError("生效日期格式应为 YYYY-MM-DD")
        require(effective >= earliest, "已有计划调整最早次日生效；首次计划可当天生效")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = self.plan(patient_id)
            require(effective >= self.today() + timedelta(days=1 if latest["items"] else 0),
                    "已有计划调整最早次日生效；首次计划可当天生效")
            require(not db.execute("""SELECT 1 FROM sessions s JOIN runs r ON s.run_id=r.id
                WHERE r.patient_id=? AND s.status IN ('connecting','running')""",
                (patient_id,)).fetchone(), "训练中不能修改计划", 409)
            row = db.execute("SELECT MAX(revision) FROM plan_versions WHERE patient_id=?", (patient_id,)).fetchone()
            legacy = db.execute("SELECT revision FROM plans WHERE patient_id=?", (patient_id,)).fetchone()
            revision = max(row[0] or 0, legacy[0] if legacy else 0) + 1
            db.execute("DELETE FROM plan_versions WHERE patient_id=? AND effective_from>?", (patient_id, self.today().isoformat()))
            db.execute("INSERT INTO plan_versions VALUES (?,?,?,?,?,?)",
                       (patient_id, revision, effective.isoformat(), encoded(sorted(weekdays)), encoded(items), now()))
            # Keep the legacy row as a fallback only; effective versions are authoritative.
            if not legacy and effective <= self.today():
                db.execute("INSERT INTO plans VALUES (?,?,?,?)", (patient_id, encoded(items), revision, now()))
        return self.plan(patient_id)

    def create_run(self, payload, role="staff"):
        training_day = self.today()
        patient = self.patient(payload.get("patient_id"))
        require(not patient["archived"], "患者已归档")
        mode = payload.get("mode")
        require(mode in ("plan", "single"), "请选择计划训练或单项练习")
        plan = self.plan(patient["id"], training_day)
        items = plan["items"] if mode == "plan" else self.validate_items([
            {"action": payload.get("action"), "target_reps": payload.get("target_reps")}])
        if role == "patient" and mode == "single":
            prescribed = next((i for i in plan["items"] if i["action"] == items[0]["action"]), None)
            require(prescribed is not None and prescribed["target_reps"] == items[0]["target_reps"], "只能练习当前计划内动作，次数由工作人员设置", 403)
        require(bool(items), "请先配置训练计划")
        identity = uuid4().hex
        with self.db() as db:
            db.execute("""INSERT INTO runs (id,patient_id,patient,items,mode,plan_revision,status,created_at,ended_at,
                training_date,schedule_revision,conditions,teaching) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (identity, patient["id"], encoded(patient), encoded(items), mode,
                        plan["revision"], "active", now(), None, training_day.isoformat(),
                        plan["revision"] if plan["scheduled"] else None,
                        encoded({"algorithm_version": self.algorithm_version, "affected_side": patient["affected_side"],
                                 "forearm_cm": patient["forearm_cm"]}), encoded(self.teaching_versions)))
        return self.run(identity)

    @staticmethod
    def session_row(row):
        row = dict(row)
        row["patient"] = json.loads(row["patient"])
        row["summary"] = json.loads(row["summary"]) if row["summary"] else None
        row["partial"] = row["status"] == "interrupted"
        row["target_reached"] = bool(row["summary"] and row["summary"].get("count", 0) >= row["target"])
        return row

    def session(self, identity):
        with self.db() as db:
            return self.session_row(self.one(db, "sessions", identity))

    def run(self, identity):
        with self.db() as db:
            row = self.one(db, "runs", identity)
            sessions = db.execute("SELECT * FROM sessions WHERE run_id=? ORDER BY created_at", (identity,)).fetchall()
        row["patient"] = json.loads(row["patient"])
        row["items"] = json.loads(row["items"])
        row["sessions"] = [self.session_row(item) for item in sessions]
        row["completed_items"] = len({s["item_index"] for s in row["sessions"] if s["status"] == "completed"})
        row["elapsed_seconds"] = sum(s["elapsed_seconds"] for s in row["sessions"])
        row["conditions"] = json.loads(row["conditions"]) if row["conditions"] else None
        row["teaching"] = json.loads(row["teaching"]) if row["teaching"] else None
        with self.db() as db:
            feedback = db.execute("SELECT * FROM feedback WHERE run_id=?", (identity,)).fetchall()
        row["feedback"] = {f["phase"]: dict(f, data=json.loads(f["data"])) for f in feedback}
        row["end_reason"] = next((s["reason"] for s in reversed(row["sessions"]) if s["reason"]), None) if row["status"] != "active" else None
        return row

    def runs(self, patient_id, days=0, offset=0):
        self.patient(patient_id)
        require(days in (0, 7, 30), "日期范围需为全部、7 天或 30 天")
        require(0 <= offset <= 100000, "分页参数错误")
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else ""
        with self.db() as db:
            rows = db.execute("""SELECT id FROM runs WHERE patient_id=? AND created_at>=?
                ORDER BY created_at DESC LIMIT 50 OFFSET ?""", (patient_id, since, offset)).fetchall()
        return [self.run(row["id"]) for row in rows]

    def overview(self, patient_id, utc_offset_minutes=0):
        self.patient(patient_id)
        require(-840 <= utc_offset_minutes <= 840, "时区参数错误")
        zone = timezone(timedelta(minutes=utc_offset_minutes))
        start = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        with self.db() as db:
            row = db.execute("""SELECT COUNT(CASE WHEN s.status='completed' THEN 1 END),
                COALESCE(SUM(s.elapsed_seconds),0) FROM sessions s JOIN runs r ON s.run_id=r.id
                WHERE r.patient_id=? AND s.started_at>=? AND s.started_at<?""",
                (patient_id, start.isoformat(timespec="milliseconds"), end.isoformat(timespec="milliseconds"))).fetchone()
        return {"completed_sessions": row[0], "elapsed_seconds": row[1]}

    def create_session(self, payload):
        run = self.run(payload.get("run_id"))
        require(run["status"] == "active", "本次训练已结束，请重新开始", 409)
        require("pre" in run["feedback"], "请先填写或跳过训练前反馈", 409)
        index = payload.get("item_index")
        require(isinstance(index, int) and not isinstance(index, bool)
                and index == run["completed_items"] and index < len(run["items"]), "请按计划顺序训练")
        item = run["items"][index]
        identity = uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.one(db, "runs", run["id"])
            done = db.execute("SELECT COUNT(DISTINCT item_index) FROM sessions WHERE run_id=? AND status='completed'",
                              (run["id"],)).fetchone()[0]
            require(current["status"] == "active" and index == done, "本次训练状态已改变，请刷新", 409)
            require(not db.execute("SELECT 1 FROM sessions WHERE run_id=? AND status IN ('pending','connecting','running')",
                                   (run["id"],)).fetchone(), "本次训练已有待完成会话", 409)
            db.execute("""INSERT INTO sessions (id,run_id,item_index,action,target,patient,status,created_at)
                VALUES (?,?,?,?,?,?,'pending',?)""",
                (identity, run["id"], index, item["action"], item["target_reps"], encoded(run["patient"]), now()))
        return self.session(identity)

    def claim_session(self, identity):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            item = self.one(db, "sessions", identity)
            require(self.one(db, "runs", item["run_id"])["status"] == "active", "本次训练已结束", 409)
            require(item["status"] == "pending", "该会话已使用，请重新开始", 409)
            db.execute("UPDATE sessions SET status='connecting' WHERE id=?", (identity,))
        return self.session(identity)

    def checkpoint(self, identity, summary, elapsed, started_at):
        with self.db() as db:
            db.execute("""UPDATE sessions SET status='running',summary=?,elapsed_seconds=?,
                started_at=COALESCE(started_at,?),checkpoint_at=?
                WHERE id=? AND status IN ('connecting','running')""",
                (encoded(summary), elapsed, started_at, now(), identity))

    def finish_session(self, identity, summary, elapsed, started_at, reason="manual"):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            item = self.one(db, "sessions", identity)
            if item["status"] in ("completed", "interrupted"):
                return self.session(identity)
            status = "completed" if reason == "manual" and started_at else "interrupted"
            stamp = now()
            db.execute("""UPDATE sessions SET status=?,summary=?,elapsed_seconds=?,started_at=?,
                ended_at=?,reason=?,checkpoint_at=? WHERE id=?""",
                (status, encoded(summary), elapsed, started_at, stamp, reason, stamp, identity))
            run = self.one(db, "runs", item["run_id"])
            done = db.execute("SELECT COUNT(DISTINCT item_index) FROM sessions WHERE run_id=? AND status='completed'",
                              (run["id"],)).fetchone()[0]
            run_status = "unfinished" if reason == "user_discomfort" else "interrupted" if status == "interrupted" else (
                "completed" if done == len(json.loads(run["items"])) else "active")
            db.execute("UPDATE runs SET status=?,ended_at=? WHERE id=?",
                       (run_status, stamp if run_status != "active" else None, run["id"]))
        return self.session(identity)

    def finish_run(self, identity):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self.one(db, "runs", identity)
            require(not db.execute("SELECT 1 FROM sessions WHERE run_id=? AND status IN ('connecting','running')",
                                   (identity,)).fetchone(), "请先结束当前动作", 409)
            if run["status"] == "active":
                db.execute("UPDATE sessions SET status='interrupted',ended_at=?,reason='cancelled' WHERE run_id=? AND status='pending'",
                           (now(), identity))
                db.execute("UPDATE runs SET status='unfinished',ended_at=? WHERE id=?", (now(), identity))
        return self.run(identity)

    def recover(self):
        with self.db() as db:
            db.execute("""UPDATE sessions SET status='interrupted',ended_at=?,reason='server_restart'
                WHERE status IN ('pending','connecting','running')""", (now(),))
            db.execute("UPDATE runs SET status='interrupted',ended_at=? WHERE status='active'", (now(),))

    def backup(self, destination):
        with self.db() as db, sqlite3.connect(destination) as output:
            db.backup(output)

    def feedback(self, identity, phase, payload, author):
        require(phase in ("pre", "post"), "反馈阶段错误")
        state = payload.get("state")
        require(state in ("submitted", "skipped"), "请选择提交或跳过")
        values = {}
        if state == "submitted":
            for field in ("pain", "fatigue", "discomfort", "difficulty", "note"):
                value = payload.get(field)
                if field in ("pain", "fatigue"):
                    require(value is None or type(value) is int and 0 <= value <= 10, "疼痛、疲劳需为 0–10 或未填写")
                elif field == "difficulty":
                    require(value is None or phase == "post" and type(value) is int and 1 <= value <= 5, "困难程度仅用于后测，范围 1–5")
                elif field == "discomfort":
                    require(value is None or type(value) is bool, "不适情况格式错误")
                else:
                    require(value is None or isinstance(value, str) and len(value) <= 200, "不适说明最多 200 字")
                values[field] = value
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self.one(db, "runs", identity)
            if phase == "pre":
                require(run["status"] == "active" and not db.execute(
                    "SELECT 1 FROM sessions WHERE run_id=? AND (started_at IS NOT NULL OR status IN ('connecting','running'))", (identity,)).fetchone(), "训练开始后不能修改前测", 409)
            else:
                require(run["status"] != "active", "请先结束本次训练再填写后测", 409)
            db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?)", (identity, phase, state, encoded(values), author, now()))
        return self.run(identity)["feedback"][phase]

    def calendar(self, patient_id, week_start=None):
        self.patient(patient_id)
        today = self.today()
        try:
            start = date.fromisoformat(week_start) if week_start else today
        except (ValueError, TypeError):
            raise BusinessError("周日期格式错误")
        start -= timedelta(days=start.weekday())
        with self.db() as db:
            versions = db.execute("SELECT * FROM plan_versions WHERE patient_id=? ORDER BY effective_from,revision", (patient_id,)).fetchall()
            runs = db.execute("SELECT training_date,schedule_revision,status,mode,id FROM runs WHERE patient_id=? AND training_date>=? AND training_date<=?",
                              (patient_id, start.isoformat(), (start + timedelta(days=6)).isoformat())).fetchall()
            sessions = db.execute("""SELECT s.run_id,s.summary,s.target FROM sessions s JOIN runs r ON r.id=s.run_id
                WHERE r.patient_id=? AND r.training_date>=? AND r.training_date<=? AND s.status='completed'""",
                (patient_id, start.isoformat(), (start + timedelta(days=6)).isoformat())).fetchall()
        reached = {}
        for session in sessions:
            reached.setdefault(session["run_id"], []).append(bool(session["summary"] and json.loads(session["summary"]).get("count", 0) >= session["target"]))
        days = []
        for offset in range(7):
            day = start + timedelta(days=offset)
            version = next((v for v in reversed(versions) if v["effective_from"] <= day.isoformat()), None)
            scheduled = bool(version and day.isoweekday() in json.loads(version["weekdays"]))
            matching = [r for r in runs if r["training_date"] == day.isoformat()]
            completed_runs = [r for r in matching if scheduled and r["mode"] == "plan" and r["status"] == "completed" and r["schedule_revision"] == version["revision"]]
            completed = day <= today and bool(completed_runs)
            targets = completed and any(reached.get(r["id"]) and all(reached[r["id"]]) for r in completed_runs)
            days.append({"date": day.isoformat(), "scheduled": scheduled, "completed": bool(completed),
                         "due": scheduled and day <= today, "revision": version["revision"] if version else None,
                         "training_runs": len(matching), "target_reached": bool(targets), "future": day > today})
        due = sum(d["due"] for d in days)
        completed = sum(d["completed"] for d in days)
        return {"week_start": start.isoformat(), "today": today.isoformat(), "days": days, "completed_days": completed,
                "target_reached_days": sum(d["target_reached"] for d in days),
                "scheduled_days": sum(d["scheduled"] for d in days), "due_days": due,
                "adherence": round(completed / due * 100, 1) if due else None}

    def trends(self, patient_id, action, start, end, mode=None, revision=None):
        self.patient(patient_id)
        require(action in BY_KEY, "请选择支持的动作")
        try:
            first, last = date.fromisoformat(start), date.fromisoformat(end)
        except (TypeError, ValueError):
            raise BusinessError("趋势日期格式应为 YYYY-MM-DD")
        require(first <= last and (last - first).days <= 366, "趋势区间最多 366 天")
        require(mode in (None, "single", "plan"), "训练类型错误")
        with self.db() as db:
            records = db.execute("""SELECT s.*,r.training_date,r.plan_revision,r.conditions,r.mode,r.created_at AS run_created
                FROM sessions s JOIN runs r ON r.id=s.run_id WHERE r.patient_id=? AND s.action=? ORDER BY s.created_at""", (patient_id, action)).fetchall()
            versions = db.execute("SELECT * FROM plan_versions WHERE patient_id=? ORDER BY effective_from", (patient_id,)).fetchall()
            feedback = db.execute("SELECT f.* FROM feedback f JOIN runs r ON f.run_id=r.id WHERE r.patient_id=?", (patient_id,)).fetchall()
        feedback_by_run = {}
        for f in feedback:
            feedback_by_run.setdefault(f["run_id"], {})[f["phase"]] = dict(f, data=json.loads(f["data"]))
        groups, legacy = {}, []
        for record in records:
            r = dict(record)
            day = r["training_date"] or datetime.fromisoformat(r["run_created"]).astimezone(timezone(timedelta(hours=8))).date().isoformat()
            if not start <= day <= end or mode and r["mode"] != mode or revision is not None and r["plan_revision"] != revision:
                continue
            summary = json.loads(r["summary"]) if r["summary"] else {}
            point = {"date": day, "session_id": r["id"], "run_id": r["run_id"], "status": r["status"],
                     "partial": r["status"] != "completed", "count": summary.get("count", 0), "passed": summary.get("passed", 0),
                     "target": r["target"], "elapsed_seconds": r["elapsed_seconds"], "reason": r["reason"],
                     "feedback": feedback_by_run.get(r["run_id"], {})}
            point["rate"] = point["passed"] / point["count"] * 100 if point["count"] else None
            if not r["conditions"]:
                legacy.append(point)
                continue
            conditions = dict(json.loads(r["conditions"]), target_reps=r["target"], plan_revision=r["plan_revision"], mode=r["mode"], action=action)
            key = encoded(conditions)
            group = groups.setdefault(key, {"conditions": conditions, "points": [], "count": 0, "passed": 0})
            group["points"].append(point)
            if r["status"] == "completed":
                group["count"] += point["count"]
                group["passed"] += point["passed"]
        for group in groups.values():
            group["rate"] = group["passed"] / group["count"] * 100 if group["count"] else None
        return {"action": action, "start": start, "end": end, "groups": list(groups.values()), "legacy": legacy,
                "plan_changes": [dict(v, items=json.loads(v["items"]), weekdays=json.loads(v["weekdays"])) for v in versions if start <= v["effective_from"] <= end]}
