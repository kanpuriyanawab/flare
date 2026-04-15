"""SQLite-backed request queue for cold-start buffering."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from flare.gateway.queue.base import BaseQueue, QueuedRequest, RequestStatus

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".flare" / "gateway.db"


def _expand_db_path(url: str) -> str:
    """Convert 'sqlite+aiosqlite:///~/.flare/gateway.db' to an absolute path string."""
    if url.startswith("sqlite+aiosqlite:///"):
        path = url[len("sqlite+aiosqlite:///"):]
        return str(Path(path).expanduser())
    return url


class SQLiteQueue(BaseQueue):
    """Async SQLite-backed queue. Safe for single-node gateway deployment."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = str(Path(str(db_path)).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS queued_requests (
                    request_id       TEXT PRIMARY KEY,
                    model_name       TEXT NOT NULL,
                    path             TEXT NOT NULL,
                    method           TEXT NOT NULL DEFAULT 'POST',
                    headers          TEXT NOT NULL DEFAULT '{}',
                    body             BLOB NOT NULL,
                    status           TEXT NOT NULL DEFAULT 'queued',
                    created_at       TEXT NOT NULL,
                    completed_at     TEXT,
                    response_status  INTEGER,
                    response_headers TEXT,
                    response_body    BLOB,
                    error            TEXT,
                    estimated_wait   INTEGER DEFAULT 300
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_status
                ON queued_requests (model_name, status)
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name       TEXT NOT NULL,
                    deployment_id    TEXT,
                    gpu_type         TEXT,
                    gpu_count        INTEGER DEFAULT 1,
                    mode             TEXT DEFAULT 'on-demand',
                    started_at       TEXT NOT NULL,
                    stopped_at       TEXT,
                    total_seconds    REAL DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash         TEXT PRIMARY KEY,
                    name             TEXT NOT NULL,
                    created_at       TEXT NOT NULL,
                    last_used_at     TEXT,
                    is_active        INTEGER DEFAULT 1
                )
            """)
            await db.commit()
        logger.info("SQLite queue initialized at %s", self._db_path)

    async def enqueue(self, request: QueuedRequest) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO queued_requests
                    (request_id, model_name, path, method, headers, body,
                     status, created_at, estimated_wait)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.model_name,
                    request.path,
                    request.method,
                    json.dumps(request.headers),
                    request.body,
                    request.status.value,
                    request.created_at.isoformat(),
                    request.estimated_wait_seconds,
                ),
            )
            await db.commit()

    async def get(self, request_id: str) -> Optional[QueuedRequest]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM queued_requests WHERE request_id = ?", (request_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return _row_to_request(row)

    async def update_status(
        self,
        request_id: str,
        status: RequestStatus,
        *,
        response_status: Optional[int] = None,
        response_headers: Optional[dict] = None,
        response_body: Optional[bytes] = None,
        error: Optional[str] = None,
    ) -> None:
        completed_at = datetime.utcnow().isoformat() if status in (
            RequestStatus.COMPLETE, RequestStatus.FAILED
        ) else None

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE queued_requests
                SET status = ?,
                    completed_at = ?,
                    response_status = ?,
                    response_headers = ?,
                    response_body = ?,
                    error = ?
                WHERE request_id = ?
                """,
                (
                    status.value,
                    completed_at,
                    response_status,
                    json.dumps(response_headers) if response_headers else None,
                    response_body,
                    error,
                    request_id,
                ),
            )
            await db.commit()

    async def list_pending(self, model_name: str) -> list[QueuedRequest]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM queued_requests
                WHERE model_name = ? AND status IN ('queued', 'waking')
                ORDER BY created_at ASC
                """,
                (model_name,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [_row_to_request(r) for r in rows]

    async def mark_model_waking(self, model_name: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE queued_requests SET status = 'waking' WHERE model_name = ? AND status = 'queued'",
                (model_name,),
            )
            await db.commit()


def _row_to_request(row: aiosqlite.Row) -> QueuedRequest:
    return QueuedRequest(
        request_id=row["request_id"],
        model_name=row["model_name"],
        path=row["path"],
        method=row["method"],
        headers=json.loads(row["headers"] or "{}"),
        body=row["body"] or b"",
        status=RequestStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        response_status=row["response_status"],
        response_headers=json.loads(row["response_headers"]) if row["response_headers"] else None,
        response_body=row["response_body"],
        error=row["error"],
        estimated_wait_seconds=row["estimated_wait"] or 300,
    )


# ---------------------------------------------------------------------------
# Cost tracking helpers (used by flare cost command)
# ---------------------------------------------------------------------------

async def record_deployment_start(
    db_path: str,
    model_name: str,
    gpu_type: str,
    gpu_count: int,
    mode: str = "on-demand",
    deployment_id: str | None = None,
) -> int:
    """Insert a cost record when a model goes RUNNING. Returns record id."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO cost_records (model_name, deployment_id, gpu_type, gpu_count, mode, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_name, deployment_id, gpu_type, gpu_count, mode, datetime.utcnow().isoformat()),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def record_deployment_stop(db_path: str, record_id: int) -> None:
    """Update cost record when a model stops."""
    now = datetime.utcnow()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT started_at FROM cost_records WHERE id = ?", (record_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            started = datetime.fromisoformat(row[0])
            total_s = (now - started).total_seconds()
            await db.execute(
                "UPDATE cost_records SET stopped_at = ?, total_seconds = ? WHERE id = ?",
                (now.isoformat(), total_s, record_id),
            )
            await db.commit()


async def get_cost_records(
    since: datetime,
    db_path: str | None = None,
    model_filter: str | None = None,
) -> list[dict]:
    """Fetch aggregated cost records since a datetime."""
    _path = db_path or str(Path.home() / ".flare" / "gateway.db")
    if not Path(_path).exists():
        return []

    query = """
        SELECT model_name, mode, gpu_type, gpu_count,
               SUM(total_seconds) as total_seconds
        FROM cost_records
        WHERE started_at >= ?
    """
    params: list = [since.isoformat()]

    if model_filter:
        query += " AND model_name = ?"
        params.append(model_filter)

    query += " GROUP BY model_name, mode, gpu_type, gpu_count ORDER BY total_seconds DESC"

    async with aiosqlite.connect(_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
