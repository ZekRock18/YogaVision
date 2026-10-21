"""Tests for session store module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogavision.session_store import SessionRecord, SessionStore


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.db"
        store = SessionStore(db_path=path)
        yield store, path


class TestSessionStore:
    def test_start_and_end_session(self, db):
        store, _ = db
        record = store.start_session(source="webcam")
        assert record.session_id
        assert record.started_at
        store.end_session(
            record.session_id,
            frames_processed=120,
            pose_counts={"tree": 80, "warrior": 40},
            duration_s=30.0,
            avg_latency_ms=15.5,
        )
        loaded = store.get_session(record.session_id)
        assert loaded is not None
        assert loaded.frames_processed == 120
        assert loaded.dominant_pose == "tree"

    def test_list_sessions(self, db):
        store, _ = db
        for i in range(3):
            r = store.start_session()
            store.end_session(r.session_id, frames_processed=i * 10, pose_counts={}, duration_s=1.0)
        sessions = store.list_sessions()
        assert len(sessions) == 3

    def test_delete_session(self, db):
        store, _ = db
        r = store.start_session()
        store.delete_session(r.session_id)
        assert store.get_session(r.session_id) is None

    def test_clear_history(self, db):
        store, _ = db
        for _ in range(3):
            r = store.start_session()
            store.end_session(r.session_id, frames_processed=1, pose_counts={}, duration_s=1.0)
        store.clear_history()
        assert store.list_sessions() == []

    def test_export_csv(self, db):
        store, _ = db
        r = store.start_session()
        store.end_session(r.session_id, frames_processed=50, pose_counts={"cobra": 50}, duration_s=10.0)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sessions.csv"
            store.export_csv(out)
            assert out.exists()
            content = out.read_text()
            assert "session_id" in content

    def test_summary_stats(self, db):
        store, _ = db
        for _ in range(2):
            r = store.start_session()
            store.end_session(r.session_id, frames_processed=100, pose_counts={"tree": 100}, duration_s=10.0)
        stats = store.summary_stats()
        assert stats["session_count"] == 2
        assert stats["total_frames"] == 200
