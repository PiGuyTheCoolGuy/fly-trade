from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from scipy import sparse

from flytrade.cli import synthetic_history
from flytrade.config import Config, Model, Risk, load_config
from flytrade.engine import Wallet, backtest, metrics
from flytrade.features import WARMUP, features, targets
from flytrade.market import Coinbase, DataError, iso, sync_range, validate_frame
from flytrade.model import FlyModel
from flytrade.research import boundaries
from flytrade.storage import CandleStore, ProcessLock, atomic_json


def candles(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.risk = Risk(initial_cash=1000, max_position_fraction=1, max_total_exposure=1,
                         risk_per_trade=1, fee_bps=0, spread_bps=0, slippage_bps=0,
                         stop_loss=0.1, take_profit=0.1, daily_loss_limit=1, max_drawdown=1)

    def test_round_trip_deducts_both_fees(self):
        risk = replace(self.risk, fee_bps=100)
        wallet = Wallet.new(1000)
        self.assertTrue(wallet.buy("BTC-USD", 100, 0, risk))
        self.assertGreaterEqual(wallet.cash, 0)
        quantity = wallet.positions["BTC-USD"].quantity
        wallet.sell("BTC-USD", 100, 300, risk)
        self.assertAlmostEqual(wallet.equity, 1000 - 2 * quantity)
        self.assertAlmostEqual(wallet.closed[0]["pnl"], wallet.equity - 1000)

    def test_cash_and_total_exposure_are_shared(self):
        risk = replace(self.risk, max_position_fraction=0.2, max_total_exposure=0.5)
        wallet = Wallet.new(1000)
        for symbol in ("A-USD", "B-USD", "C-USD", "D-USD"):
            wallet.marks[symbol] = 100
            wallet.buy(symbol, 100, 0, risk)
        self.assertAlmostEqual(wallet.exposure, 500)
        self.assertAlmostEqual(wallet.cash, 500)
        self.assertEqual(len(wallet.positions), 3)

    def test_no_short_and_no_double_buy(self):
        wallet = Wallet.new(1000)
        self.assertFalse(wallet.sell("BTC-USD", 100, 0, self.risk))
        self.assertTrue(wallet.buy("BTC-USD", 100, 0, self.risk))
        self.assertFalse(wallet.buy("BTC-USD", 100, 0, self.risk))

    def test_signal_fills_at_next_open(self):
        frame = candles([(0, 100, 100, 100, 100, 1), (300, 120, 120, 120, 120, 1), (600, 120, 120, 120, 120, 1)])
        wallet = backtest({"BTC-USD": frame}, {"BTC-USD": pd.Series({0: 0.2})},
                          {"BTC-USD": 0.01}, self.risk, 300, 0, 900)
        self.assertEqual(wallet.fills[0]["timestamp"], 300)
        self.assertEqual(wallet.fills[0]["price"], 120)

    def test_same_bar_stop_wins_over_take(self):
        frame = candles([(0, 100, 100, 100, 100, 1), (300, 100, 130, 80, 100, 1)])
        wallet = backtest({"BTC-USD": frame}, {"BTC-USD": pd.Series({0: 0.2})},
                          {"BTC-USD": 0.01}, self.risk, 300, 0, 600)
        self.assertEqual(wallet.fills[1]["reason"], "stop_loss")
        self.assertAlmostEqual(wallet.fills[1]["price"], 90)

    def test_stop_gap_uses_worse_open(self):
        frame = candles([(0, 100, 100, 100, 100, 1), (300, 100, 100, 100, 100, 1), (600, 70, 75, 65, 70, 1)])
        wallet = backtest({"BTC-USD": frame}, {"BTC-USD": pd.Series({0: 0.2, 300: 0.2})},
                          {"BTC-USD": 0.01}, self.risk, 300, 0, 900)
        self.assertEqual(wallet.fills[1]["price"], 70)
        self.assertEqual(wallet.fills[1]["reason"], "stop_gap")
        self.assertEqual(len(wallet.fills), 2)

    def test_loss_halts_latch_and_daily_resets(self):
        risk = replace(self.risk, max_drawdown=0.1, daily_loss_limit=0.03)
        wallet = Wallet.new(1000)
        wallet.begin_day(0)
        wallet.cash = 890
        self.assertTrue(wallet.risk_check(risk))
        self.assertTrue(wallet.halted)
        self.assertTrue(wallet.daily_halted)
        wallet.begin_day(86400)
        self.assertFalse(wallet.daily_halted)
        self.assertTrue(wallet.halted)
        self.assertFalse(wallet.buy("BTC-USD", 100, 86400, risk))

    def test_wallet_serialization_preserves_accounting(self):
        wallet = Wallet.new(1000)
        wallet.buy("BTC-USD", 100, 0, self.risk)
        restored = Wallet.deserialize(json.loads(json.dumps(wallet.serialize())))
        self.assertEqual(restored.serialize(), wallet.serialize())
        self.assertEqual(restored.equity, wallet.equity)

    def test_max_hold_and_final_liquidation(self):
        risk = replace(self.risk, max_hold_bars=1)
        frame = candles([(t, 100, 100, 100, 100, 1) for t in (0, 300, 600)])
        wallet = backtest({"BTC-USD": frame}, {"BTC-USD": pd.Series({0: .2, 300: .2})},
                          {"BTC-USD": .01}, risk, 300, 0, 900)
        self.assertEqual(wallet.fills[-1]["reason"], "max_hold")
        self.assertFalse(wallet.positions)
        self.assertEqual(metrics(wallet)["closed_trades"], 1)


class FeatureModelTests(unittest.TestCase):
    def test_future_changes_cannot_change_past_features(self):
        frame = synthetic_history(Config(), 500)["BTC-USD"]
        original = features(frame)
        changed = frame.copy()
        changed.iloc[350:, :4] *= 10
        pd.testing.assert_frame_equal(original.iloc[:350], features(changed).iloc[:350])

    def test_online_window_matches_training_features(self):
        frame = synthetic_history(Config(), 500)["BTC-USD"]
        np.testing.assert_allclose(features(frame).iloc[-1], features(frame.iloc[-216:]).iloc[-1], atol=1e-12)

    def test_labels_and_purge_stay_before_boundary(self):
        config = Config()
        frame = synthetic_history(config, 1000)["BTC-USD"]
        y = targets(frame, config.model.horizon)
        self.assertAlmostEqual(y.iloc[100], np.log(frame.open.iloc[113] / frame.open.iloc[101]))
        fit_end, train_end, _ = boundaries(1000, config)
        self.assertLess((fit_end - 1) + config.model.horizon + 1, train_end)
        self.assertTrue(y.iloc[-13:].isna().all())

    def test_model_fit_save_load_and_train_only_scaling(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 17))
        y = x[:, 0] * 0.02
        model = FlyModel(Model(hidden_size=32), 17)
        model.fit(x, y)
        np.testing.assert_allclose(model.mean, x.mean(axis=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = FlyModel.load(path)
            np.testing.assert_allclose(model.predict(x), loaded.predict(x))
        self.assertLess(np.mean((model.predict(x) - y) ** 2), 1e-4)

    def test_real_graph_transform_is_deterministic_and_uses_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            graph = sparse.csr_matrix(np.roll(np.eye(20), 1, axis=0) * .8)
            sparse.save_npz(path, graph)
            model = FlyModel(Model(kind="flywire", hidden_size=8, brain_steps=3), 5, path)
            x = np.random.default_rng(2).normal(size=(130, 5))
            model.fit(x, x[:, 1] * .01)
            encoded = model.encode(x)
            np.testing.assert_array_equal(encoded, model.encode(x))
            model.graph *= 0
            self.assertGreater(np.max(np.abs(encoded - model.encode(x))), 0.01)


class MarketStorageTests(unittest.TestCase):
    def test_coinbase_order_sort_filter_and_candle_validation(self):
        client = Coinbase(Config())
        with patch.object(client, "get", return_value=[[600, 9, 12, 10, 11, 5], [300, 9, 12, 10, 11, 5],
                                                        [0, 9, 12, 10, 11, 5], [300, 9, 12, 10, 11, 5]]):
            self.assertEqual(client.candles("BTC-USD", 300, 0, 600), [(0, 10., 12., 9., 11., 5.), (300, 10., 12., 9., 11., 5.)])

    def test_bad_prices_and_gaps_are_rejected(self):
        for frame in [candles([(0, 10, 9, 8, 10, 2)]), candles([(0, float("nan"), 12, 8, 10, 2)]),
                      candles([(0, 10, 12, 8, 10, 2), (600, 10, 12, 8, 10, 2)])]:
            with self.assertRaises(DataError):
                validate_frame(frame, 300)

    def test_download_paginates_and_resumes_cached_pages(self):
        class Fake:
            calls = []
            def candles(self, symbol, timeframe, start, end):
                self.calls.append((start, end))
                return [(t, 100, 101, 99, 100, 10) for t in range(start, end, timeframe)]
        with tempfile.TemporaryDirectory() as directory:
            store = CandleStore(Path(directory) / "test.sqlite3")
            client = Fake()
            frame = sync_range(client, store, "BTC-USD", 300, 0, 700 * 300)
            self.assertEqual(len(frame), 700)
            self.assertEqual(len(client.calls), 3)
            self.assertTrue(all((b-a)//300 <= 299 for a,b in client.calls))
            sync_range(client, store, "BTC-USD", 300, 0, 700 * 300)
            self.assertEqual(len(client.calls), 3)

    def test_incomplete_download_does_not_fabricate_candles(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CandleStore(Path(directory) / "test.sqlite3")
            client = Coinbase(Config())
            with patch.object(client, "candles", return_value=[(0, 100, 101, 99, 100, 10)]):
                with self.assertRaises(DataError):
                    sync_range(client, store, "BTC-USD", 300, 0, 600)
            self.assertEqual(len(store.read("BTC-USD", 300, 0, 600)), 1)

    def test_stale_crossed_auction_quotes_rejected(self):
        client = Coinbase(Config())
        now = 1700000000
        for raw in [{"bids": [[100]], "asks": [[101]], "time": iso(now - 100)},
                    {"bids": [[102]], "asks": [[101]], "time": iso(now)},
                    {"bids": [[100]], "asks": [[101]], "time": iso(now), "auction_mode": True}]:
            with patch.object(client, "get", return_value=raw), self.assertRaises(DataError):
                client.quote("BTC-USD", now)
        with patch.object(client, "get", return_value={"bids": [[100]], "asks": [[101]], "time": iso(now)}):
            self.assertEqual(client.quote("BTC-USD", now)["mid"], 100.5)

    def test_rate_limit_retries_get_only(self):
        client = Coinbase(Config())
        error = HTTPError("url", 429, "slow down", {"Retry-After": "1"}, None)
        with patch("flytrade.market.urlopen", side_effect=[error, io.BytesIO(b'[]')]) as opened, patch("flytrade.market.time.sleep"):
            self.assertEqual(client.get("/products/BTC-USD/candles"), [])
            self.assertEqual(opened.call_count, 2)
            self.assertTrue(all(call.args[0].method == "GET" for call in opened.call_args_list))

    def test_config_rejects_unknown_real_money_and_invalid_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for content in ['[live]\nenabled = true\n', '[risk]\nfee_bps = nan\n', '[model]\nhorizon = 1.5\n']:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_atomic_json_and_exclusive_process_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.json"
            atomic_json(path, {"cash": 12})
            atomic_json(path, {"cash": 13})
            self.assertEqual(json.loads(path.read_text())["cash"], 13)
            lock_path = Path(directory) / "process.lock"
            with ProcessLock(lock_path):
                with self.assertRaises(RuntimeError), ProcessLock(lock_path):
                    pass
            with ProcessLock(lock_path):
                pass


if __name__ == "__main__":
    unittest.main()
