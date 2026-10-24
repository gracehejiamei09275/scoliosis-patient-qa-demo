from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PASSWORD_ITERATIONS = 310_000
SESSION_DAYS = 14


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat().replace("+00:00", "Z")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


class AccountStore:
    """Small account and assessment store supporting SQLite locally and Postgres on Render."""

    def __init__(self, database_url: str | None, *, sqlite_path: Path | None = None) -> None:
        self.database_url = (database_url or "").strip()
        self.sqlite_path = sqlite_path or Path("outputs") / "patient_accounts.sqlite3"
        self.is_postgres = self.database_url.startswith(("postgres://", "postgresql://"))
        self.admin_emails = {
            value.strip().lower()
            for value in os.environ.get("ADMIN_EMAILS", "").split(",")
            if value.strip()
        }
        if not self.is_postgres:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:  # pragma: no cover - only relevant in deployed Postgres mode
                raise RuntimeError("Postgres mode requires psycopg[binary].") from exc
            connection = psycopg.connect(self.database_url, row_factory=dict_row)
        else:
            connection = sqlite3.connect(self.sqlite_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def _p(self) -> str:
        return "%s" if self.is_postgres else "?"

    @staticmethod
    def _dict(row: Any | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def initialize(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                consent_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                patient_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                prediction_json TEXT,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_assessments_user_created ON assessments(user_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)",
        ]
        with self._connect() as connection:
            for statement in statements:
                connection.execute(statement)

    def _public_user(self, user: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_login_at": user.get("last_login_at"),
        }

    def register(self, *, email: str, password: str, display_name: str, consent: bool) -> tuple[dict[str, Any], str]:
        email = email.strip().lower()
        display_name = display_name.strip()
        if not EMAIL_RE.match(email):
            raise ValueError("请输入有效的邮箱地址。")
        if len(password) < 10:
            raise ValueError("密码至少需要 10 个字符。")
        if not display_name or len(display_name) > 80:
            raise ValueError("姓名或昵称不能为空，且不能超过 80 个字符。")
        if not consent:
            raise ValueError("注册前需要同意账号与健康数据存储说明。")
        user_id = str(uuid.uuid4())
        created_at = _iso()
        role = "admin" if email in self.admin_emails else "user"
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO users (id,email,password_hash,display_name,role,consent_at,created_at) VALUES ({','.join([self._p] * 7)})",
                    (user_id, email, _hash_password(password), display_name, role, created_at, created_at),
                )
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                raise ValueError("该邮箱已经注册，请直接登录。") from exc
            raise
        return self.login(email=email, password=password)

    def login(self, *, email: str, password: str) -> tuple[dict[str, Any], str]:
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM users WHERE email={self._p}", (email,)
            ).fetchone()
            user = self._dict(row)
            if user is None or not _verify_password(password, user["password_hash"]):
                raise ValueError("邮箱或密码不正确。")
            desired_role = "admin" if email in self.admin_emails else user["role"]
            logged_at = _iso()
            connection.execute(
                f"UPDATE users SET role={self._p}, last_login_at={self._p} WHERE id={self._p}",
                (desired_role, logged_at, user["id"]),
            )
            user["role"] = desired_role
            user["last_login_at"] = logged_at
            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
            expires_at = _iso(_now() + timedelta(days=SESSION_DAYS))
            connection.execute(
                f"INSERT INTO sessions (token_hash,user_id,created_at,expires_at) VALUES ({','.join([self._p] * 4)})",
                (token_hash, user["id"], logged_at, expires_at),
            )
        return self._public_user(user), raw_token

    def user_for_token(self, raw_token: str | None) -> dict[str, Any] | None:
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT users.* FROM sessions
                JOIN users ON users.id=sessions.user_id
                WHERE sessions.token_hash={self._p} AND sessions.expires_at>{self._p}
                """,
                (token_hash, _iso()),
            ).fetchone()
        user = self._dict(row)
        return self._public_user(user) if user else None

    def logout(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute(f"DELETE FROM sessions WHERE token_hash={self._p}", (token_hash,))

    def save_assessment(
        self,
        *,
        user_id: str,
        question: str,
        patient: dict[str, Any],
        answer: str,
        prediction: dict[str, Any] | None,
    ) -> str:
        assessment_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO assessments (id,user_id,question,patient_json,answer,prediction_json,created_at) VALUES ({','.join([self._p] * 7)})",
                (
                    assessment_id,
                    user_id,
                    question,
                    json.dumps(patient, ensure_ascii=False),
                    answer,
                    json.dumps(prediction, ensure_ascii=False) if prediction is not None else None,
                    _iso(),
                ),
            )
        return assessment_id

    @staticmethod
    def _assessment(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "question": row["question"],
            "patient": json.loads(row["patient_json"]),
            "answer": row["answer"],
            "prediction": json.loads(row["prediction_json"]) if row.get("prediction_json") else None,
            "created_at": row["created_at"],
        }

    def history(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM assessments WHERE user_id={self._p} ORDER BY created_at DESC LIMIT {self._p}",
                (user_id, min(max(limit, 1), 200)),
            ).fetchall()
        return [self._assessment(dict(row)) for row in rows]

    def admin_stats(self) -> dict[str, Any]:
        seven_days_ago = _iso(_now() - timedelta(days=7))
        with self._connect() as connection:
            registered = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
            assessments = connection.execute("SELECT COUNT(*) AS count FROM assessments").fetchone()["count"]
            active = connection.execute(
                f"SELECT COUNT(DISTINCT user_id) AS count FROM assessments WHERE created_at>={self._p}",
                (seven_days_ago,),
            ).fetchone()["count"]
        return {"registered_users": registered, "assessments": assessments, "active_users_7d": active}

    def admin_users(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT users.id, users.email, users.display_name, users.role, users.created_at,
                       users.last_login_at, COUNT(assessments.id) AS assessment_count,
                       MAX(assessments.created_at) AS last_assessment_at
                FROM users LEFT JOIN assessments ON assessments.user_id=users.id
                GROUP BY users.id, users.email, users.display_name, users.role, users.created_at, users.last_login_at
                ORDER BY users.created_at DESC LIMIT {self._p}
                """,
                (min(max(limit, 1), 500),),
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_assessments(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT assessments.*, users.email, users.display_name
                FROM assessments JOIN users ON users.id=assessments.user_id
                ORDER BY assessments.created_at DESC LIMIT {self._p}
                """,
                (min(max(limit, 1), 500),),
            ).fetchall()
        results = []
        for row in rows:
            item = self._assessment(dict(row))
            item["email"] = row["email"]
            item["display_name"] = row["display_name"]
            results.append(item)
        return results

