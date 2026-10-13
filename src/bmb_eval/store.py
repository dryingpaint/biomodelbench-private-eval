"""Node-side job queue, result release and ladder state (sqlite, state/node.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    submitter TEXT NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done
    created REAL NOT NULL,
    release_at REAL NOT NULL,      -- result hidden until this time
    finished REAL,
    result TEXT,                   -- exported payload (already gated)
    posted INTEGER NOT NULL DEFAULT 0  -- result delivered to the hub
);
CREATE INDEX IF NOT EXISTS by_quota ON submissions(submitter, task, day);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS ladder (
    submitter TEXT NOT NULL,
    task TEXT NOT NULL,
    contract_version INTEGER NOT NULL,
    score REAL NOT NULL,
    metrics TEXT NOT NULL,
    PRIMARY KEY (submitter, task, contract_version)
);
"""


class Store:
    def __init__(self, state_dir: Path):
        self.db = sqlite3.connect(state_dir / "node.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(submissions)")}
        if "split" not in cols:   # nodes installed before held-out test sets
            self.db.execute("ALTER TABLE submissions ADD COLUMN split TEXT NOT NULL DEFAULT 'validation'")
            self.db.commit()
        self.lock = threading.Lock()

    def add(self, sid: str, task: str, submitter: str, release_after_s: int,
            split: str = "validation") -> None:
        now = time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO submissions (id, task, submitter, day, status, created, "
                "release_at, finished, result, posted, split) VALUES (?,?,?,?,?,?,?,?,?,0,?)",
                (sid, task, submitter, _day(), "queued", now, now + release_after_s, None, None, split))

    def count_today(self, submitter: str, task: str) -> int:
        with self.lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM submissions WHERE submitter=? AND task=? AND day=? "
                "AND split='validation'", (submitter, task, _day())).fetchone()[0]

    def count_total(self, task: str) -> int:
        with self.lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM submissions WHERE task=? AND split='validation'",
                (task,)).fetchone()[0]

    def count_final(self, task: str) -> int:
        """Final (test set) evaluations run or accepted for this task, ever."""
        with self.lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM submissions WHERE task=? AND split='test'",
                (task,)).fetchone()[0]

    def next_queued(self) -> tuple[str, str, str] | None:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT id, task, submitter FROM submissions WHERE status='queued' "
                "ORDER BY created LIMIT 1").fetchone()
            if row:
                self.db.execute("UPDATE submissions SET status='running' WHERE id=?", (row[0],))
            return row

    def start(self, sid: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE submissions SET status='running' WHERE id=?", (sid,))

    def finish(self, sid: str, result: dict) -> None:
        with self.lock, self.db:
            self.db.execute(
                "UPDATE submissions SET status='done', finished=?, result=? WHERE id=?",
                (time.time(), json.dumps(result), sid))

    def has(self, sid: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM submissions WHERE id=?", (sid,)).fetchone() is not None

    def released_unposted(self) -> list[tuple[str, dict]]:
        """Finished results whose release time has passed and that the hub
        has not received yet. Holding them here, on the provider's side, is
        what keeps run time from being observable."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, result FROM submissions WHERE status='done' AND posted=0 "
                "AND release_at <= ? ORDER BY created", (time.time(),)).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def unposted(self) -> list[str]:
        """Every job this node holds whose result the hub hasn't received yet."""
        with self.lock:
            return [r[0] for r in self.db.execute(
                "SELECT id FROM submissions WHERE posted=0").fetchall()]

    def mark_posted(self, sid: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE submissions SET posted=1 WHERE id=?", (sid,))

    def requeue_running(self) -> int:
        """On restart, runs interrupted mid-flight go back to the queue."""
        with self.lock, self.db:
            return self.db.execute(
                "UPDATE submissions SET status='queued' WHERE status='running'").rowcount

    def summary(self) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT task, status, COUNT(*) FROM submissions GROUP BY task, status").fetchall()
        out: dict = {}
        for task, status, n in rows:
            out.setdefault(task, {})[status] = n
        return out

    def bump(self, name: str, reset: bool = False) -> int:
        with self.lock, self.db:
            if reset:
                self.db.execute("INSERT OR REPLACE INTO counters VALUES (?, 0)", (name,))
                return 0
            self.db.execute(
                "INSERT INTO counters VALUES (?, 1) "
                "ON CONFLICT(name) DO UPDATE SET value = value + 1", (name,))
            return self.db.execute("SELECT value FROM counters WHERE name=?",
                                   (name,)).fetchone()[0]

    # ---------------------------------------------------------------- ladder

    def ladder_best(self, submitter: str, task: str, version: int) -> tuple[float, dict] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT score, metrics FROM ladder WHERE submitter=? AND task=? "
                "AND contract_version=?", (submitter, task, version)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def ladder_set(self, submitter: str, task: str, version: int, score: float,
                   metrics: dict) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO ladder VALUES (?,?,?,?,?)",
                            (submitter, task, version, score, json.dumps(metrics)))


def _day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())
