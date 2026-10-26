"""Local two-role authentication and real SMTP verification mail."""

from datetime import datetime, timezone
import hashlib
import hmac
import os
import re
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from uuid import uuid4

from rehab_service.storage.database import BusinessError, now, require


def normalize_email(value):
    require(isinstance(value, str), "请输入有效邮箱")
    value = value.strip().lower()
    require(len(value) <= 254 and re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value), "请输入有效邮箱")
    return value


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def password_hash(value, salt=None):
    salt = salt or secrets.token_hex(16)
    result = hashlib.pbkdf2_hmac("sha256", value.encode(), salt.encode(), 260000).hex()
    return salt + ":" + result


def normalize_username(value):
    require(isinstance(value, str), "请输入患者账号")
    value = value.strip().lower()
    require(re.fullmatch(r"[a-z0-9_.-]{3,32}", value), "账号需为 3–32 位字母、数字、下划线、点或短横线")
    return value


def validate_password(value):
    require(isinstance(value, str) and 8 <= len(value) <= 128 and bool(value.strip()), "密码需为 8–128 个字符")
    return value


def send_mail(email, code, purpose="login"):
    host, sender, password = (os.getenv(key, "") for key in ("SMTP_HOST", "SMTP_FROM", "SMTP_PASSWORD"))
    require(host and sender and password, "邮件服务尚未配置，请联系工作人员", 503)
    security = os.getenv("SMTP_SECURITY", "starttls").lower()
    require(security in ("ssl", "starttls"), "SMTP_SECURITY 仅支持 ssl 或 starttls", 503)
    try:
        port = int(os.getenv("SMTP_PORT", "465" if security == "ssl" else "587"))
        message = EmailMessage()
        label = "密码重置" if purpose == "reset" else "登录"
        message["Subject"] = "康复训练" + label + "验证码"
        message["From"], message["To"] = sender, email
        message.set_content("您的康复训练" + label + "验证码为：" + code + "\n五分钟内有效。请勿向他人透露。如非本人操作，请忽略此邮件。")
        context = ssl.create_default_context()
        smtp_type = smtplib.SMTP_SSL if security == "ssl" else smtplib.SMTP
        kwargs = {"context": context} if security == "ssl" else {}
        with smtp_type(host, port, timeout=10, **kwargs) as smtp:
            if security == "starttls":
                smtp.starttls(context=context)
            smtp.login(os.getenv("SMTP_USER", sender), password)
            smtp.send_message(message)
    except (OSError, ValueError, smtplib.SMTPException) as exc:
        raise BusinessError("验证码邮件发送失败，请稍后重试或联系工作人员", 503) from exc


class AuthService:
    def __init__(self, store):
        self.store = store
        with store.db() as db:
            if not db.execute("SELECT 1 FROM accounts WHERE username='admin'").fetchone():
                db.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?)",
                           (uuid4().hex, "staff", "admin", None, password_hash("123"), now()))
        self.dummy_password = password_hash(secrets.token_urlsafe(24))

    def patient_account(self, patient_id):
        self.store.patient(patient_id)
        with self.store.db() as db:
            row = db.execute("SELECT username,password FROM accounts WHERE patient_id=?", (patient_id,)).fetchone()
        return {"patient_id": patient_id, "username": row["username"] if row else None,
                "password_configured": bool(row and row["password"])}

    def save_patient_account(self, patient_id, payload):
        username = normalize_username(payload.get("username"))
        password = payload.get("password")
        hashed = password_hash(validate_password(password)) if password is not None else None
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            self.store.one(db, "patients", patient_id)
            account = db.execute("SELECT * FROM accounts WHERE patient_id=?", (patient_id,)).fetchone()
            require(not db.execute("SELECT 1 FROM accounts WHERE username=? AND (patient_id IS NULL OR patient_id<>?)",
                                   (username, patient_id)).fetchone(), "该账号已被使用", 409)
            require(hashed or (account and account["password"]), "首次分配账号时请设置密码")
            require(not db.execute("""SELECT 1 FROM sessions s JOIN runs r ON s.run_id=r.id
                WHERE r.patient_id=? AND s.status IN ('connecting','running')""", (patient_id,)).fetchone(),
                    "训练中不能修改患者账号", 409)
            if account:
                db.execute("UPDATE accounts SET username=?,password=? WHERE id=?",
                           (username, hashed or account["password"], account["id"]))
                if hashed or username != account["username"]:
                    db.execute("DELETE FROM login_sessions WHERE account_id=?", (account["id"],))
                    db.execute("DELETE FROM password_reset_codes WHERE account_id=?", (account["id"],))
                    db.execute("DELETE FROM email_codes WHERE email=(SELECT email FROM patients WHERE id=?)", (patient_id,))
            else:
                db.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?)",
                           (uuid4().hex, "patient", username, patient_id, hashed, now()))
        return self.patient_account(patient_id)

    def patient_login(self, payload, ip):
        self.limit("patient-login-ip:" + ip, 30, 300)
        username = normalize_username(payload.get("username"))
        self.limit("patient-login-account:" + username, 10, 300)
        password = payload.get("password")
        require(isinstance(password, str) and len(password) <= 128, "账号或密码错误", 401)
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            account = db.execute("""SELECT a.* FROM accounts a JOIN patients p ON a.patient_id=p.id
                WHERE a.username=? AND a.role='patient' AND p.archived=0""", (username,)).fetchone()
            saved = account["password"] if account and account["password"] else self.dummy_password
            valid = hmac.compare_digest(saved, password_hash(password, saved.split(":")[0]))
            require(account and account["password"] and valid, "账号或密码错误", 401)
            return self.issue(account, db)

    def password_reset_code(self, payload, ip):
        username = normalize_username(payload.get("username"))
        email = normalize_email(payload.get("email"))
        self.limit("reset-send-ip:" + ip, 20, 3600)
        self.limit("reset-send-email:" + email, 1, 60)
        with self.store.db() as db:
            account = db.execute("""SELECT a.* FROM accounts a JOIN patients p ON a.patient_id=p.id
                WHERE a.username=? AND a.role='patient' AND p.email=? AND p.archived=0""", (username, email)).fetchone()
        require(account and account["password"], "账号与绑定邮箱不匹配，请核对或联系管理员", 400)
        code = str(secrets.randbelow(1000000)).zfill(6)
        send_mail(email, code, "reset")
        salt = secrets.token_hex(16)
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("""SELECT a.* FROM accounts a JOIN patients p ON a.patient_id=p.id
                WHERE a.id=? AND a.username=? AND p.email=? AND p.archived=0""", (account["id"], username, email)).fetchone()
            require(current and current["password"] == account["password"], "账号或邮箱已变化，请重新获取验证码", 409)
            stamp = time.time()
            db.execute("INSERT OR REPLACE INTO password_reset_codes VALUES (?,?,?,?,?,?)",
                       (email, account["id"], salt + ":" + digest(salt + code), stamp + 300, 0, stamp))
        return {"sent": True, "retry_after": 60, "expires_in": 300}

    def password_reset(self, payload, ip):
        username = normalize_username(payload.get("username"))
        email = normalize_email(payload.get("email"))
        password = validate_password(payload.get("password"))
        code = payload.get("code")
        require(isinstance(code, str) and re.fullmatch(r"\d{6}", code), "请输入六位验证码")
        self.limit("reset-ip:" + ip, 30, 3600)
        self.limit("reset-email:" + email, 20, 3600)
        hashed = password_hash(password)
        valid = False
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM password_reset_codes WHERE email=?", (email,)).fetchone()
            account = db.execute("""SELECT a.* FROM accounts a JOIN patients p ON a.patient_id=p.id
                WHERE a.username=? AND a.role='patient' AND p.email=? AND p.archived=0""", (username, email)).fetchone()
            if row and row["expires"] > time.time() and row["attempts"] < 5:
                salt, saved = row["digest"].split(":")
                valid = bool(account and account["id"] == row["account_id"] and hmac.compare_digest(saved, digest(salt + code)))
                if not valid:
                    db.execute("UPDATE password_reset_codes SET attempts=attempts+1 WHERE email=?", (email,))
                else:
                    db.execute("UPDATE accounts SET password=? WHERE id=?", (hashed, account["id"]))
                    db.execute("DELETE FROM login_sessions WHERE account_id=?", (account["id"],))
                    db.execute("DELETE FROM password_reset_codes WHERE email=?", (email,))
                    db.execute("DELETE FROM email_codes WHERE email=?", (email,))
        require(valid, "验证码错误、已失效或尝试次数过多", 401)
        return {"reset": True}

    def limit(self, bucket, count, seconds):
        stamp = time.time()
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM auth_limits WHERE at<?", (stamp - 86400,))
            used = db.execute("SELECT COUNT(*) FROM auth_limits WHERE bucket=? AND at>?",
                              (bucket, stamp - seconds)).fetchone()[0]
            require(used < count, "操作过于频繁，请稍后再试", 429)
            db.execute("INSERT INTO auth_limits VALUES (?,?)", (bucket, stamp))

    def email_code(self, payload, ip):
        email = normalize_email(payload.get("email"))
        self.limit("send-ip:" + ip, 20, 3600)
        self.limit("send-email:" + email, 1, 60)
        with self.store.db() as db:
            bound = db.execute("SELECT id FROM patients WHERE email=?", (email,)).fetchone()
        require(bound, "该邮箱尚未绑定患者，请联系工作人员", 400)
        code = str(secrets.randbelow(1000000)).zfill(6)
        # Only replace a valid old challenge after SMTP accepted the new message.
        send_mail(email, code)
        stamp = time.time()
        salt = secrets.token_hex(16)
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT id FROM patients WHERE email=?", (email,)).fetchone()
            require(current and current["id"] == bound["id"], "邮箱绑定已变化，请重新获取验证码", 409)
            db.execute("INSERT OR REPLACE INTO email_codes VALUES (?,?,?,?,?)",
                       (email, salt + ":" + digest(salt + code), stamp + 300, 0, stamp))
        return {"sent": True, "retry_after": 60, "expires_in": 300}

    def issue(self, account, db=None):
        if db is None:
            with self.store.db() as connection:
                return self.issue(account, connection)
        token = secrets.token_urlsafe(32)
        ttl = 8 * 3600 if account["role"] == "staff" else 24 * 3600
        expires = time.time() + ttl
        db.execute("DELETE FROM login_sessions WHERE expires<=?", (time.time(),))
        db.execute("INSERT INTO login_sessions VALUES (?,?,?,?)", (digest(token), account["id"], expires, now()))
        return {"token": token, "expires": expires, "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
                "role": account["role"], "patient_id": account["patient_id"]}

    def email_login(self, payload, ip):
        email = normalize_email(payload.get("email"))
        self.limit("login-ip:" + ip, 60, 3600)
        self.limit("login-email:" + email, 20, 3600)
        code = payload.get("code")
        require(isinstance(code, str) and re.fullmatch(r"\d{6}", code), "请输入六位验证码")
        valid = False
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM email_codes WHERE email=?", (email,)).fetchone()
            if row and row["expires"] > time.time() and row["attempts"] < 5:
                salt, saved = row["digest"].split(":")
                valid = hmac.compare_digest(saved, digest(salt + code))
                if valid:
                    db.execute("DELETE FROM email_codes WHERE email=?", (email,))
                else:
                    db.execute("UPDATE email_codes SET attempts=attempts+1 WHERE email=?", (email,))
            patient = db.execute("SELECT id FROM patients WHERE email=?", (email,)).fetchone()
            if valid and patient:
                account = db.execute("SELECT * FROM accounts WHERE patient_id=?", (patient["id"],)).fetchone()
                if not account:
                    identity = uuid4().hex
                    db.execute("INSERT INTO accounts VALUES (?, 'patient', NULL, ?, NULL, ?)",
                               (identity, patient["id"], now()))
                    account = db.execute("SELECT * FROM accounts WHERE id=?", (identity,)).fetchone()
                return self.issue(account, db)
        require(valid and patient, "验证码错误、已失效或尝试次数过多", 401)

    def staff_login(self, payload, ip):
        self.limit("staff-ip:" + ip, 10, 300)
        with self.store.db() as db:
            account = db.execute("SELECT * FROM accounts WHERE username='admin'").fetchone()
        password = payload.get("password", "")
        valid = isinstance(password, str) and len(password) <= 256 and payload.get("username") == "admin"
        if valid:
            valid = hmac.compare_digest(account["password"], password_hash(password, account["password"].split(":")[0]))
        require(valid, "账号或密码错误", 401)
        return self.issue(account)

    def user(self, authorization):
        require(isinstance(authorization, str) and authorization.startswith("Bearer "), "请先登录", 401)
        token_hash = digest(authorization[7:])
        with self.store.db() as db:
            row = db.execute("""SELECT a.id,a.role,a.patient_id,s.expires FROM login_sessions s
                JOIN accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires>?""",
                (token_hash, time.time())).fetchone()
        require(row, "登录已失效，请重新登录", 401)
        return dict(row, token_hash=token_hash)

    def logout(self, user):
        with self.store.db() as db:
            db.execute("DELETE FROM login_sessions WHERE token_hash=?", (user["token_hash"],))
        return {"logged_out": True}


def current_user(request):
    return request.app.state.auth.user(request.headers.get("authorization"))


def staff(user):
    require(user["role"] == "staff", "该操作仅限工作人员", 403)


def patient_access(user, patient_id):
    require(user["role"] == "staff" or user["patient_id"] == patient_id, "不能访问其他患者的数据", 403)


def run_access(request, user, run_id):
    record = request.app.state.store.run(run_id)
    patient_access(user, record["patient_id"])
    return record


def session_access(request, user, session_id):
    record = request.app.state.store.session(session_id)
    run_access(request, user, record["run_id"])
    return record
