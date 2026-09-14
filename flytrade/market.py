"""Unauthenticated, GET-only Coinbase Exchange market data."""

from datetime import datetime, timezone
import json
import logging
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import Config
from .storage import CandleStore

LOG = logging.getLogger(__name__)
BASE = "https://api.exchange.coinbase.com"


class DataError(RuntimeError):
    pass


def iso(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _quote_timestamp(value: str) -> float:
    """Parse exchange timestamps consistently on Python 3.10 and newer."""
    if not isinstance(value, str):
        raise ValueError("quote timestamp must be a string")
    match = re.fullmatch(
        r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
        r"(?:\.([0-9]+))?(Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])", value)
    if match is None:
        raise ValueError("quote timestamp must be ISO 8601 with a timezone")
    whole, fraction, offset = match.groups()
    # Coinbase sends nanoseconds; datetime stores microseconds. Python 3.10
    # requires fractional seconds to have exactly three or six digits.
    fraction = "." + fraction[:6].ljust(6, "0") if fraction else ""
    offset = "+00:00" if offset == "Z" else offset
    return datetime.fromisoformat(whole + fraction + offset).timestamp()


def validate_frame(frame: pd.DataFrame, timeframe: int, *, continuous: bool = True) -> None:
    if frame.empty:
        raise DataError("No candles returned")
    idx = frame.index.to_numpy(dtype=np.int64)
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise DataError("Non-finite, nonpositive prices, or negative volume in candle data")
    if (np.diff(idx) <= 0).any() or (idx % timeframe != 0).any():
        raise DataError("Candle timestamps must be ordered, unique, and aligned")
    if ((frame.low > frame[["open", "close", "high"]].min(axis=1)).any()
            or (frame.high < frame[["open", "close", "low"]].max(axis=1)).any()):
        raise DataError("Invalid candle high/low bounds")
    if continuous and (np.diff(idx) != timeframe).any():
        raise DataError("Candle history has gaps; retry download or choose a liquid pair/shorter history")


class Coinbase:
    def __init__(self, config: Config):
        self.config = config
        self.last_request = 0.0

    def get(self, path: str, params: dict | None = None):
        url = BASE + path + ("?" + urlencode(params) if params else "")
        for attempt in range(5):
            delay = self.config.market.request_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()
            try:
                req = Request(url, headers={"User-Agent": "FlyTrade/0.1 public-market-research",
                                            "Accept": "application/json"}, method="GET")
                with urlopen(req, timeout=self.config.market.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise DataError(f"Coinbase HTTP {exc.code} for {path}; check symbol/network availability") from exc
                retry = exc.headers.get("Retry-After", "0")
                try:
                    delay = min(30, max(2 ** attempt, float(retry)))
                except ValueError:
                    delay = 2 ** attempt
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                delay = 2 ** attempt
                if attempt == 4:
                    raise DataError(f"Market data unavailable: {exc}. Cached data is retained; retry later.") from exc
            if attempt < 4:
                LOG.warning("Retrying market data in %.1fs (attempt %d/5)", delay, attempt + 2)
                time.sleep(delay)
        raise DataError(f"Coinbase temporarily unavailable for {path}")

    def candles(self, symbol: str, timeframe: int, start: int, end: int) -> list[tuple]:
        raw = self.get(f"/products/{quote(symbol, safe='')}/candles",
                       {"start": iso(start), "end": iso(end), "granularity": timeframe})
        if not isinstance(raw, list):
            raise DataError("Unexpected candle response from Coinbase")
        rows = {}
        try:
            for row in raw:
                if len(row) != 6 or float(row[0]) != int(row[0]):
                    raise ValueError("Candle must contain six values and an integer timestamp")
                ts, low, high, opening, close, volume = row
                if start <= int(ts) < end:
                    rows[int(ts)] = (int(ts), float(opening), float(high), float(low), float(close), float(volume))
            values = sorted(rows.values())
            if values:
                frame = pd.DataFrame(values, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
                validate_frame(frame, timeframe, continuous=False)
            return values
        except (TypeError, ValueError, IndexError) as exc:
            raise DataError(f"Malformed candles: {exc}") from exc

    def quote(self, symbol: str, now: float | None = None) -> dict:
        # Level-1 book timestamp describes the quote; ticker time is last trade time.
        raw = self.get(f"/products/{quote(symbol, safe='')}/book", {"level": 1})
        now = time.time() if now is None else now
        try:
            if raw.get("auction_mode", False):
                raise ValueError("indicative auction quotes are not executable")
            bid, ask = float(raw["bids"][0][0]), float(raw["asks"][0][0])
            ts = _quote_timestamp(raw["time"])
            if not all(map(math.isfinite, [bid, ask, ts])) or not 0 < bid <= ask:
                raise ValueError("invalid bid/ask")
            if not -5 <= now - ts <= self.config.app.quote_max_age_seconds:
                raise ValueError("stale or future quote timestamp")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DataError(f"Unusable {symbol} quote: {exc}") from exc
        return {"bid": bid, "ask": ask, "timestamp": ts, "mid": (bid + ask) / 2}


def sync_range(client: Coinbase, store: CandleStore, symbol: str, timeframe: int,
               start: int, end: int) -> pd.DataFrame:
    """Persist each page; subsequent runs only request incomplete pages."""
    start, end = int(start), int(end)
    if start >= end or start % timeframe or end % timeframe:
        raise ValueError("Data range must be aligned and nonempty")
    for cursor in range(start, end, 299 * timeframe):
        page_end = min(end, cursor + 299 * timeframe)
        expected = (page_end - cursor) // timeframe
        if len(store.read(symbol, timeframe, cursor, page_end)) == expected:
            continue
        rows = client.candles(symbol, timeframe, cursor, page_end)
        store.put(symbol, timeframe, rows)
        LOG.info("%s history %.0f%%", symbol, 100 * (page_end - start) / (end - start))
    frame = store.read(symbol, timeframe, start, end)
    expected = (end - start) // timeframe
    if len(frame) != expected:
        raise DataError(f"{symbol}: {len(frame)}/{expected} candles cached. Missing candles are not invented. "
                        "Retry download; if persistent, shorten history_days or remove that symbol.")
    validate_frame(frame, timeframe)
    return frame


def download_history(config: Config, client: Coinbase | None = None, end: int | None = None) -> dict:
    client = client or Coinbase(config)
    tf = config.market.timeframe
    end = (int(time.time()) if end is None else end) // tf * tf
    start = (end - config.market.history_days * 86400) // tf * tf
    store = CandleStore(config.data_dir / "candles.sqlite3")
    frames = {}
    for symbol in config.market.symbols:
        frames[symbol] = sync_range(client, store, symbol, tf, start, end)
    return frames
