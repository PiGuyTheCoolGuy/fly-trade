from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

from flytrade.cli import reproduce_test, synthetic_history
from flytrade.config import App, Config, Market, Model
from flytrade.dashboard import start_dashboard
from flytrade.features import features
from flytrade.market import DataError
from flytrade.model import FlyModel
from flytrade.paper import PaperRunner
from flytrade.research import run_research
from flytrade.storage import atomic_json


class FakeMarket:
    def __init__(self):
        self.bid = 100.0
        self.ask = 100.1
        self.stale = False
        self.fail = False

    def candles(self, symbol, timeframe, start, end):
        if self.fail:
            raise DataError("Test network outage")
        return [(ts, 100., 100.2, 99.8, 100., 10.) for ts in range(start, end, timeframe)]

    def quote(self, symbol, now):
        if self.fail:
            raise DataError("Test network outage")
        return {"bid": self.bid, "ask": self.ask, "mid": (self.bid + self.ask) / 2,
                "timestamp": now - 3600 if self.stale else now}


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Config(root=Path(self.tmp.name), market=Market(symbols=["BTC-USD"]), model=Model(hidden_size=32))
        self.client = FakeMarket()
        self.now = 1705000200 + 10
        self.run_id = "test-fixture"
        directory = self.config.data_dir / "runs" / self.run_id
        frame = synthetic_history(self.config, 500)["BTC-USD"]
        x = features(frame).dropna()
        model = FlyModel(self.config.model, x.shape[1])
        model.fit(x.to_numpy(), np.full(len(x), .05))
        model.metadata["feature_names"] = list(x.columns)
        path = directory / "BTC-USD-paper.npz"
        model.save(path)
        # An explicit test artifact; it is never published or used as evidence.
        self.report = {"run_id": self.run_id, "synthetic": False,
                       "fingerprint": self.config.trading_fingerprint(), "thresholds": {"BTC-USD": .01},
                       "paper_model_sha256": {"BTC-USD": hashlib.sha256(path.read_bytes()).hexdigest()},
                       "paper_training_end": 1704067200}
        atomic_json(directory / "report.json", self.report)
        atomic_json(self.config.data_dir / "latest.json", {"run_id": self.run_id})

    def runner(self):
        runner = PaperRunner(self.config, self.client)
        runner.models["BTC-USD"].predict = lambda _: np.array([.05])
        return runner

    def test_restart_and_same_candle_do_not_duplicate_orders(self):
        runner = self.runner()
        runner.step(self.now)
        self.assertEqual(len(runner.wallet.fills), 1)
        restored = self.runner()
        restored.step(self.now + 15)
        self.assertEqual(len(restored.wallet.fills), 1)
        self.assertEqual(restored.wallet.cash, runner.wallet.cash)

    def test_paper_uses_observed_quote_not_historical_open(self):
        self.client.ask = 110
        self.client.bid = 109.9
        runner = self.runner()
        runner.step(self.now)
        self.assertAlmostEqual(runner.wallet.fills[0]["price"], 110 * 1.0005)

    def test_stop_is_checked_between_candles_and_cannot_reenter_same_signal(self):
        runner = self.runner()
        runner.step(self.now)
        self.client.bid, self.client.ask = 95, 95.1
        runner.step(self.now + 15)
        self.assertEqual(len(runner.wallet.fills), 2)
        self.assertEqual(runner.wallet.fills[-1]["side"], "SELL")
        restored = self.runner()
        restored.step(self.now + 30)
        self.assertEqual(len(restored.wallet.fills), 2)

    def test_stale_quotes_cannot_mutate_wallet(self):
        runner = self.runner()
        original = runner.wallet.serialize()
        self.client.stale = True
        with self.assertRaises(DataError):
            runner.step(self.now)
        self.assertEqual(runner.wallet.serialize(), original)

    def test_outage_is_saved_and_once_returns_error(self):
        runner = self.runner()
        self.client.fail = True
        with self.assertRaises(DataError):
            runner.run(Event(), once=True)
        state = json.loads((self.config.data_dir / "paper.json").read_text())
        self.assertEqual(state["wallet"]["fills"], [])
        self.assertIn("outage", state["last_error"])

    def test_reconnect_only_consumes_latest_signal(self):
        runner = self.runner()
        runner.step(self.now)
        runner.step(self.now + 2 * 86400)
        self.assertEqual(len(runner.wallet.fills), 2)  # previous position expires; no catch-up trades
        self.assertEqual(runner.state["last_signal"]["BTC-USD"], int(self.now + 2 * 86400) // 300 * 300 - 300)

    def test_changed_risk_and_model_tampering_are_rejected(self):
        self.runner().step(self.now)
        changed = replace(self.config, risk=replace(self.config.risk, initial_cash=20000))
        with self.assertRaises(ValueError):
            PaperRunner(changed, self.client)
        path = self.config.data_dir / "runs" / self.run_id / "BTC-USD-paper.npz"
        path.write_bytes(path.read_bytes() + b"modified")
        with self.assertRaises(ValueError):
            self.runner()

    def test_synthetic_models_cannot_start_paper(self):
        self.report["synthetic"] = True
        atomic_json(self.config.data_dir / "runs" / self.run_id / "report.json", self.report)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            self.runner()

    def test_interrupted_tick_cannot_save_partial_wallet(self):
        runner = self.runner()
        original = runner.wallet.serialize()
        buy = runner.wallet.buy
        def interrupt_after_buy(*args, **kwargs):
            buy(*args, **kwargs)
            raise KeyboardInterrupt()
        with patch.object(runner.wallet, "buy", side_effect=interrupt_after_buy):
            with self.assertRaises(KeyboardInterrupt):
                runner.step(self.now)
        self.assertEqual(runner.wallet.serialize(), original)
        restored = self.runner()
        restored.step(self.now)
        self.assertEqual(len(restored.wallet.fills), 1)


class EndToEndTests(unittest.TestCase):
    def test_train_test_reproduce_and_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(root=Path(directory), market=Market(symbols=["BTC-USD"]),
                            model=Model(hidden_size=32), app=App(dashboard_port=0))
            frames = synthetic_history(config, 1200)
            report = run_research(config, frames, synthetic=True)
            self.assertTrue(report["synthetic"])
            self.assertNotEqual(report["ranges"]["train_end_exclusive"], report["ranges"]["test_start"])
            self.assertEqual(reproduce_test(config)["closed_trades"], report["test"]["closed_trades"])
            with self.assertRaises(ValueError):
                PaperRunner(config)
            server = start_dashboard(config)
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/api/status") as response:
                    status = json.load(response)
                self.assertEqual(status["report"]["run_id"], report["run_id"])
                with urlopen(f"http://127.0.0.1:{port}/") as response:
                    self.assertIn(b"SIMULATED MONEY", response.read())
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
