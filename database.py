"""
database.py
-----------
Persistence layer with three tiers, in order of preference:
  1. Firebase Firestore (cloud, source of truth when reachable)
  2. SQLite (local disk cache — survives process restarts on edge devices)
  3. In-memory ring buffer (last resort if disk is unavailable, e.g. read-only rootfs)

Writes always land locally first (SQLite/ring buffer), then get opportunistically
synced to Firestore by a background task. This means the ingestion path never
blocks on — or fails because of — a flaky internet connection on IoT/edge hardware.
"""

import asyncio
import json
import logging
import sqlite3
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional

from app.config import get_settings
from app.schemas import ThreatEvent

logger = logging.getLogger("sentinel.database")
settings = get_settings()

_FIRESTORE_AVAILABLE = False
try:
    if settings.FIREBASE_ENABLED:
        import firebase_admin
        from firebase_admin import credentials, firestore

        _FIRESTORE_AVAILABLE = True
except ImportError:
    logger.warning("firebase_admin not installed; Firestore sync disabled, using local persistence only.")


class LocalCache:
    """SQLite-backed cache with an in-memory ring buffer fallback."""

    def __init__(self, db_path: str, ring_buffer_size: int):
        self._ring_buffer: Deque[Dict] = deque(maxlen=ring_buffer_size)
        self._sqlite_ok = True
        self.db_path = db_path

        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS threat_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        payload TEXT NOT NULL,
                        synced INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            # e.g. read-only filesystem on a locked-down IoT gateway
            logger.warning("SQLite unavailable (%s); falling back to in-memory ring buffer only.", exc)
            self._sqlite_ok = False

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save(self, event: ThreatEvent) -> None:
        payload = event.model_dump_json()
        if self._sqlite_ok:
            try:
                with self._connect() as conn:
                    conn.execute("INSERT INTO threat_events (payload) VALUES (?)", (payload,))
                    conn.commit()
                    return
            except sqlite3.OperationalError as exc:
                logger.error("SQLite write failed (%s); falling back to ring buffer.", exc)
                self._sqlite_ok = False
        # Fallback path
        self._ring_buffer.append(json.loads(payload))

    def get_unsynced(self, limit: int = 100) -> List[Dict]:
        if self._sqlite_ok:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, payload FROM threat_events WHERE synced = 0 ORDER BY id LIMIT ?", (limit,)
                ).fetchall()
                return [{"row_id": r[0], "payload": json.loads(r[1])} for r in rows]
        return [{"row_id": None, "payload": item} for item in list(self._ring_buffer)[:limit]]

    def mark_synced(self, row_ids: List[int]) -> None:
        if self._sqlite_ok and row_ids:
            with self._connect() as conn:
                conn.executemany("UPDATE threat_events SET synced = 1 WHERE id = ?", [(r,) for r in row_ids])
                conn.commit()

    def get_recent(self, limit: int = 50) -> List[Dict]:
        if self._sqlite_ok:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT payload FROM threat_events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [json.loads(r[0]) for r in rows]
        return list(self._ring_buffer)[-limit:][::-1]


class FirestoreSync:
    """Thin async wrapper around the Firestore client, used opportunistically."""

    def __init__(self):
        self.enabled = settings.FIREBASE_ENABLED and _FIRESTORE_AVAILABLE
        self._client = None
        if self.enabled:
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                    firebase_admin.initialize_app(cred, {"projectId": settings.FIREBASE_PROJECT_ID})
                self._client = firestore.client()
                logger.info("Firestore client initialized.")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to initialize Firestore (%s); operating in offline mode.", exc)
                self.enabled = False

    def push_event(self, event_dict: Dict) -> bool:
        """Synchronous Firestore write — call via asyncio.to_thread from async code."""
        if not self.enabled or self._client is None:
            return False
        try:
            self._client.collection("threat_events").add(event_dict)
            return True
        except Exception as exc:  # noqa: BLE001 — network errors, quota, auth expiry, etc.
            logger.warning("Firestore push failed (%s); will retry from local cache.", exc)
            return False


class PersistenceManager:
    """
    Public facade used by API routes. Handles the write-local-then-sync-remote flow
    and exposes read helpers for the /threats and /metrics endpoints.
    """

    def __init__(self):
        self.cache = LocalCache(settings.SQLITE_PATH, settings.RING_BUFFER_SIZE)
        self.firestore = FirestoreSync()
        self._sync_task: Optional[asyncio.Task] = None
        self.total_threats_detected = 0

    async def record_threat(self, event: ThreatEvent) -> None:
        """Always succeeds locally first; Firestore sync happens out-of-band."""
        self.cache.save(event)
        self.total_threats_detected += 1

    def get_recent_threats(self, limit: int = 50) -> List[Dict]:
        return self.cache.get_recent(limit)

    def start_background_sync(self) -> None:
        if self._sync_task is None and self.firestore.enabled:
            self._sync_task = asyncio.create_task(self._sync_loop(), name="firestore-sync-loop")
            logger.info("Firestore background sync loop started.")

    async def stop_background_sync(self) -> None:
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        while True:
            try:
                pending = self.cache.get_unsynced(limit=100)
                synced_ids = []
                for item in pending:
                    ok = await asyncio.to_thread(self.firestore.push_event, item["payload"])
                    if ok and item["row_id"] is not None:
                        synced_ids.append(item["row_id"])
                if synced_ids:
                    self.cache.mark_synced(synced_ids)
                    logger.debug("Synced %d events to Firestore.", len(synced_ids))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never let the sync loop die
                logger.exception("Sync loop error: %s", exc)
            await asyncio.sleep(settings.SYNC_RETRY_SECONDS)


# Module-level singleton
persistence = PersistenceManager()
