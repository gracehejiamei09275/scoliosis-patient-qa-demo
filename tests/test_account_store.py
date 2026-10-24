from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pipeline.account_store import AccountStore


def test_account_lifecycle_and_admin_data() -> None:
    previous_admins = os.environ.get("ADMIN_EMAILS")
    os.environ["ADMIN_EMAILS"] = "owner@example.com"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AccountStore(None, sqlite_path=Path(temp_dir) / "accounts.sqlite3")
            user, token = store.register(
                email="owner@example.com",
                password="correct-horse-battery-staple",
                display_name="项目管理员",
                consent=True,
            )
            assert user["role"] == "admin"
            assert store.user_for_token(token)["id"] == user["id"]

            assessment_id = store.save_assessment(
                user_id=user["id"],
                question="大概可能改善多少？",
                patient={"age": 13, "pre_cobb": 28},
                answer="探索性回答",
                prediction={"predicted_cobb_change": 5.2},
            )
            history = store.history(user["id"])
            assert history[0]["id"] == assessment_id
            assert history[0]["patient"]["pre_cobb"] == 28
            assert store.admin_stats() == {
                "registered_users": 1,
                "assessments": 1,
                "active_users_7d": 1,
            }
            assert store.admin_users()[0]["assessment_count"] == 1
            assert store.admin_assessments()[0]["email"] == "owner@example.com"

            store.logout(token)
            assert store.user_for_token(token) is None
    finally:
        if previous_admins is None:
            os.environ.pop("ADMIN_EMAILS", None)
        else:
            os.environ["ADMIN_EMAILS"] = previous_admins


def test_rejects_weak_password_and_missing_consent() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = AccountStore(None, sqlite_path=Path(temp_dir) / "accounts.sqlite3")
        try:
            store.register(email="person@example.com", password="short", display_name="用户", consent=True)
            raise AssertionError("weak password should fail")
        except ValueError as exc:
            assert "10" in str(exc)
        try:
            store.register(
                email="person@example.com",
                password="long-enough-password",
                display_name="用户",
                consent=False,
            )
            raise AssertionError("missing consent should fail")
        except ValueError as exc:
            assert "同意" in str(exc)


def main() -> None:
    test_account_lifecycle_and_admin_data()
    test_rejects_weak_password_and_missing_consent()
    print("All account store tests passed.")


if __name__ == "__main__":
    main()

