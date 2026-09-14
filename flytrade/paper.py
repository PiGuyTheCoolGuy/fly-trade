"""Forward paper trading at fresh observed bid/ask prices; no real orders."""

import hashlib
import json
import logging
import math
from pathlib import Path
import time

from .config import Config
from .connectome import graph_path
from .engine import Wallet, metrics
from .features import WARMUP, features
from .market import Coinbase, DataError, iso, sync_range
from .model import FlyModel
from .research import load_report
from .storage import CandleStore, atomic_json

LOG = logging.getLogger(__name__)


class PaperRunner:
    def __init__(self, config: Config, client: Coinbase | None = None):
        self.config = config
        self.client = client or Coinbase(config)
        self.path = config.data_dir / "paper.json"
        self.store = CandleStore(config.data_dir / "candles.sqlite3")
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None
        self.report = load_report(config, self.state["run_id"] if self.state else None)
        if self.report["synthetic"]:
            raise ValueError("Synthetic demo models cannot start forward paper trading. Run train with real history first.")
        fingerprint = config.trading_fingerprint()
        if self.report["fingerprint"] != fingerprint or (self.state and self.state["fingerprint"] != fingerprint):
            raise ValueError("Trading settings changed. Use a new app.data_dir for a separate experiment; existing wallet is preserved.")
        self.wallet = Wallet.deserialize(self.state["wallet"]) if self.state else Wallet.new(config.risk.initial_cash)
        if self.state is None:
            self.state = {"schema_version": 1, "mode": "paper", "run_id": self.report["run_id"],
                          "fingerprint": fingerprint, "created_at": iso(time.time()),
                          "last_signal": {}, "signals": {}, "last_error": None,
                          "last_poll": None, "status": "starting"}
        self.models = {}
        for symbol in config.market.symbols:
            path = config.data_dir / "runs" / self.report["run_id"] / f"{symbol}-paper.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.report["paper_model_sha256"][symbol]:
                raise ValueError(f"{symbol} model changed since this experiment was created")
            self.models[symbol] = FlyModel.load(path, graph_path(config) if config.model.kind == "flywire" else None)
        self.save()

    def save(self):
        self.state["wallet"] = self.wallet.serialize()
        self.state["metrics"] = metrics(self.wallet)
        atomic_json(self.path, self.state)

    def step(self, now: float | None = None) -> dict:
        try:
            return self._step(now)
        except BaseException:
            # If an interrupt/error occurs partway through a tick, restore the
            # last committed snapshot. Never save half a tick in run()'s finally.
            self.state = json.loads(self.path.read_text())
            self.wallet = Wallet.deserialize(self.state["wallet"])
            raise

    def _step(self, now: float | None = None) -> dict:
        clock = time.time if now is None else lambda: now
        tf, risk = self.config.market.timeframe, self.config.risk
        end = int(clock()) // tf * tf
        predictions, quotes = {}, {}
        # Get complete, recent market observations before mutating the wallet.
        # On reconnect only the latest signal is eligible; old bars never fill.
        for symbol in self.config.market.symbols:
            frame = sync_range(self.client, self.store, symbol, tf, end - (WARMUP + 120) * tf, end)
            signal_ts = int(frame.index[-1])
            if signal_ts < self.report["paper_training_end"]:
                raise DataError("Market clock predates this model's training data")
            x = features(frame)
            if list(x.columns) != self.models[symbol].metadata["feature_names"]:
                raise ValueError("Feature schema changed; retrain")
            prediction = float(self.models[symbol].predict(x.iloc[-1:].to_numpy())[0])
            if not math.isfinite(prediction):
                raise DataError("Non-finite model prediction")
            predictions[symbol] = (signal_ts, prediction)
            quotes[symbol] = self.client.quote(symbol, clock())
        observed_at = clock()
        for symbol, q in quotes.items():
            if not -5 <= observed_at - q["timestamp"] <= self.config.app.quote_max_age_seconds:
                raise DataError(f"{symbol} quote expired during market refresh")
            if observed_at - (predictions[symbol][0] + tf) > tf:
                raise DataError("Candle signal expired during market refresh")
        ts = int(observed_at)
        self.wallet.begin_day(ts)
        self.wallet.marks.update({s: q["mid"] for s, q in quotes.items()})
        halted = self.wallet.risk_check(risk)
        exited = set()
        before = len(self.wallet.fills)
        slip = risk.slippage_bps / 10000
        for symbol, pos in list(self.wallet.positions.items()):
            signal_ts, prediction = predictions[symbol]
            new_signal = signal_ts > self.state["last_signal"].get(symbol, -1)
            bid = quotes[symbol]["bid"]
            reason = None
            if halted:
                reason = "risk_limit"
            elif bid <= pos.stop:
                reason = "observed_stop"
            elif bid >= pos.take:
                reason = "observed_take"
            elif ts - pos.opened_at >= risk.max_hold_bars * tf:
                reason = "max_hold"
            elif new_signal and prediction <= 0:
                reason = "model_exit"
            if reason:
                self.wallet.sell(symbol, bid * (1 - slip), ts, risk, reason)
                exited.add(symbol)
        for symbol in self.config.market.symbols:
            signal_ts, prediction = predictions[symbol]
            new_signal = signal_ts > self.state["last_signal"].get(symbol, -1)
            threshold = self.report["thresholds"][symbol]
            if new_signal and symbol not in exited and prediction > threshold:
                self.wallet.buy(symbol, quotes[symbol]["ask"] * (1 + slip), ts, risk)
            self.state["last_signal"][symbol] = signal_ts
            self.state["signals"][symbol] = {"timestamp": signal_ts, "prediction": prediction,
                                                "threshold": threshold, "quote": quotes[symbol],
                                                "decision": "entry signal" if prediction > threshold else "hold / cash"}
        if self.wallet.risk_check(risk):
            for symbol in list(self.wallet.positions):
                self.wallet.sell(symbol, quotes[symbol]["bid"] * (1 - slip), ts, risk, "risk_limit")
        # One sample per candle bounds curve growth while wallet/fills remain exact.
        self.wallet.record(end)
        self.state.update({"last_poll": iso(observed_at), "last_error": None,
                           "status": "halted" if self.wallet.halted else "daily pause" if self.wallet.daily_halted else "running"})
        # The wallet, fills and consumed signal IDs commit together. A restart
        # cannot repeat a buy from an already-committed signal.
        self.save()
        for fill in self.wallet.fills[before:]:
            LOG.info("PAPER %s %s %.8f @ %.4f (%s)", fill["side"], fill["symbol"],
                     fill["quantity"], fill["price"], fill["reason"])
        LOG.info("PAPER equity $%.2f | cash $%.2f | %d positions | %s",
                 self.wallet.equity, self.wallet.cash, len(self.wallet.positions), self.state["status"])
        return self.state

    def run(self, stop_event, *, once: bool = False):
        try:
            while not stop_event.is_set():
                try:
                    self.step()
                except DataError as exc:
                    self.state.update({"status": "waiting for market data", "last_error": str(exc)})
                    self.save()
                    LOG.error("%s", exc)
                    if once:
                        raise
                if once:
                    return
                stop_event.wait(self.config.market.poll_seconds)
        finally:
            # Stopping the process does not pretend an exit was executed.
            self.state["status"] = "stopped"
            self.save()
