"""Session persistence with SQLite."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Settings, get_settings
from .utils import get_logger

logger = get_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL NOT NULL DEFAULT 0,
    frames_processed INTEGER NOT NULL DEFAULT 0,
    dominant_pose TEXT,
    pose_counts TEXT NOT NULL DEFAULT '{}',
    avg_latency_ms REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'webcam',
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""


@dataclass
class SessionRecord:
    session_id: str
    started_at: str
    ended_at: str | None = None
    duration_s: float = 0.0
    frames_processed: int = 0
    dominant_pose: str | None = None
    pose_counts: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    source: str = "webcam"
    notes: str | None = None


class SessionStore:
    """SQLite-backed session history store."""

    def __init__(self, db_path: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = Path(db_path) if db_path else self.settings.session_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.executescript(SCHEMA)
            yield cur
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._cursor() as cur:
            cur.executescript(SCHEMA)

    # -- write ------------------------------------------------------------
    def start_session(self, source: str = "webcam") -> SessionRecord:
        record = SessionRecord(
            session_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        )
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, started_at, source) VALUES (?, ?, ?)",
                (record.session_id, record.started_at, record.source),
            )
        return record

    def end_session(
        self,
        session_id: str,
        *,
        frames_processed: int,
        pose_counts: dict[str, int],
        duration_s: float,
        avg_latency_ms: float = 0.0,
        dominant_pose: str | None = None,
    ) -> None:
        dominant = dominant_pose
        if dominant is None and pose_counts:
            dominant = max(pose_counts.items(), key=lambda kv: kv[1])[0]
        ended = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """UPDATE sessions SET ended_at=?, duration_s=?, frames_processed=?,
                   dominant_pose=?, pose_counts=?, avg_latency_ms=? WHERE session_id=?""",
                (
                    ended,
                    float(duration_s),
                    int(frames_processed),
                    dominant,
                    json.dumps(pose_counts),
                    float(avg_latency_ms),
                    session_id,
                ),
            )

    def delete_session(self, session_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    def clear_history(self) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM sessions")

    # -- read -------------------------------------------------------------
    def list_sessions(self, limit: int = 100) -> list[SessionRecord]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def _row_to_record(self, row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_s=row["duration_s"],
            frames_processed=row["frames_processed"],
            dominant_pose=row["dominant_pose"],
            pose_counts=json.loads(row["pose_counts"] or "{}"),
            avg_latency_ms=row["avg_latency_ms"],
            source=row["source"],
            notes=row["notes"],
        )

    def export_csv(self, output_path: str | Path) -> Path:
        import pandas as pd

        records = self.list_sessions(limit=10_000)
        rows = [
            {
                "session_id": r.session_id,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "duration_s": round(r.duration_s, 2),
                "frames_processed": r.frames_processed,
                "dominant_pose": r.dominant_pose,
                "avg_latency_ms": round(r.avg_latency_ms, 2),
                "source": r.source,
                "pose_counts": json.dumps(r.pose_counts),
            }
            for r in records
        ]
        df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out

    def summary_stats(self) -> dict[str, Any]:
        sessions = self.list_sessions(limit=10_000)
        total_frames = sum(s.frames_processed for s in sessions)
        dominant: dict[str, int] = {}
        for s in sessions:
            for k, v in s.pose_counts.items():
                dominant[k] = dominant.get(k, 0) + v
        return {
            "session_count": len(sessions),
            "total_frames": total_frames,
            "last_session": sessions[0].started_at if sessions else None,
            "aggregate_pose_counts": dominant,
        }
