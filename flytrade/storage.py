"""SQLite candles and atomic, inspectable JSON artifacts."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class CandleStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT, timeframe INTEGER, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY(symbol, timeframe, timestamp))""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, symbol: str, timeframe: int, rows: list[tuple]) -> None:
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)",
                           [(symbol, timeframe, *row) for row in rows])

    def read(self, symbol: str, timeframe: int, start: int, end: int) -> pd.DataFrame:
        with self.connect() as db:
            df = pd.read_sql_query("""SELECT timestamp, open, high, low, close, volume
                FROM candles WHERE symbol=? AND timeframe=? AND timestamp>=? AND timestamp<?
                ORDER BY timestamp""", db, params=(symbol, timeframe, int(start), int(end)))
        return df.set_index("timestamp")


class ProcessLock:
    """OS-level lock: released by the OS after a crash; works on Windows/Linux."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")

    def __enter__(self):
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            # Inspect file size without reading a byte that another Windows
            # process may already have locked.
            if os.fstat(self.handle.fileno()).st_size == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self.handle.close()
                raise RuntimeError("Another Fly Trade process is using this data directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.handle.close()
                raise RuntimeError("Another Fly Trade process is using this data directory") from exc
        return self

    def __exit__(self, *args):
        self.handle.close()
