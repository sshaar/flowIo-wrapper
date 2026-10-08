"""Append-only JSONL storage, one file per session.

Each line is one event::

    {"v": 1, "session_id": "...", "seq": 0, "t": "2026-10-07T16:40:00.123Z",
     "type": "cell_edit", "data": {...}}

Lines are flushed and fsync'd as they are written, so a crash of the host
application loses at most the event in flight. Readers skip a truncated
final line instead of failing.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .schema import EVENT_TYPES, SCHEMA_VERSION, SchemaError


class SessionWriter:
    def __init__(self, path: Path, session_id: str, start_seq: int = 0):
        self.path = Path(path)
        self.session_id = session_id
        self._seq = start_seq
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, type_: str, data: Dict[str, Any], t: str) -> Dict[str, Any]:
        if type_ not in EVENT_TYPES:
            raise SchemaError(f"unknown event type {type_!r}")
        with self._lock:
            if self._fh.closed:
                raise RuntimeError(f"session {self.session_id} is closed")
            rec = {"v": SCHEMA_VERSION, "session_id": self.session_id,
                   "seq": self._seq, "t": t, "type": type_, "data": data}
            self._fh.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._seq += 1
            return rec

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class Store:
    def __init__(self, root: os.PathLike):
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        if not session_id or "/" in session_id or "\\" in session_id or session_id.startswith("."):
            raise SchemaError(f"invalid session id {session_id!r}")
        return self.sessions_dir / f"{session_id}.jsonl"

    def open_writer(self, session_id: str) -> SessionWriter:
        path = self.path_for(session_id)
        start = 0
        if path.exists():
            _truncate_torn_tail(path)
            events = read_events(path)
            start = max((e["seq"] for e in events), default=-1) + 1
        return SessionWriter(path, session_id, start_seq=start)

    def session_ids(self) -> List[str]:
        return sorted(p.stem for p in self.sessions_dir.glob("*.jsonl"))

    def read(self, session_id: str) -> List[Dict[str, Any]]:
        return read_events(self.path_for(session_id))

    def iter_sessions(self) -> Iterator[List[Dict[str, Any]]]:
        for sid in self.session_ids():
            events = self.read(sid)
            if events:
                yield events


def _truncate_torn_tail(path: Path) -> None:
    """Drop a partial final line (crash mid-write) so appends start clean."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        os.truncate(path, data.rfind(b"\n") + 1)


def read_events(path: Path) -> List[Dict[str, Any]]:
    """Valid events sorted by seq. Corrupt lines are skipped; the resulting
    seq gaps are reported by ``dataset.replay`` rather than raised here, so
    one bad line does not discard a whole session."""
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("seq"), int) or "type" not in rec:
            continue
        events.append(rec)
    events.sort(key=lambda e: e["seq"])
    return events


def last_event(events: List[Dict[str, Any]], type_: str) -> Optional[Dict[str, Any]]:
    for e in reversed(events):
        if e["type"] == type_:
            return e
    return None
