"""One-command entry point for the complete experiment."""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import signal
import sys
from threading import Event

import numpy as np
import pandas as pd

from .config import Config, Market, Model, Risk, load_config
from .connectome import prepare_graph
from .dashboard import snapshot, start_dashboard
from .engine import backtest, metrics
from .features import features
from .market import download_history
from .model import FlyModel
from .paper import PaperRunner
from .research import load_report, run_research
from .storage import ProcessLock, atomic_json

LOG = logging.getLogger(__name__)


def synthetic_history(config: Config, bars: int = 4000) -> dict:
    """Deterministic fake prices for testing software, never investment evidence."""
    tf = config.market.timeframe
    start = 1704067200 // tf * tf  # January 2024, explicitly synthetic.
    index = pd.Index(start + np.arange(bars) * tf, name="timestamp")
    rng = np.random.default_rng(config.model.seed)
    frames = {}
    for i, symbol in enumerate(config.market.symbols):
        drift = np.where((np.arange(bars) // 160) % 2 == 0, 0.0020, -0.0017)
        returns = drift + rng.normal(0, 0.0025, bars)
        close = (100 + 50 * i) * np.exp(np.cumsum(returns))
        opening = np.r_[close[0] / np.exp(returns[0]), close[:-1]]
        wick = rng.uniform(0.0002, 0.002, bars)
        frames[symbol] = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * (1 + wick),
                                       "low": np.minimum(opening, close) * (1 - wick), "close": close,
                                       "volume": rng.lognormal(6, 0.7, bars)}, index=index)
    return frames


def reproduce_test(config: Config, run_id: str | None = None) -> dict:
    report = load_report(config, run_id)
    directory = config.data_dir / "runs" / report["run_id"]
    settings = report["config"]
    frozen = Config(market=Market(**settings["market"]), model=Model(**settings["model"]), risk=Risk(**settings["risk"]))
    frames, predictions = {}, {}
    start = int(pd.Timestamp(report["ranges"]["test_start"]).timestamp())
    end = int(pd.Timestamp(report["ranges"]["test_end_exclusive"]).timestamp())
    for symbol in report["symbols"]:
        frames[symbol] = pd.read_csv(directory / f"{symbol}-candles.csv.gz", index_col="timestamp", float_precision="round_trip")
        model = FlyModel.load(directory / f"{symbol}-evaluation.npz",
                              config.data_dir / "brain" / f"flywire-{frozen.model.brain_neurons}.npz"
                              if frozen.model.kind == "flywire" else None)
        x = features(frames[symbol]).loc[start:]
        predictions[symbol] = pd.Series(model.predict(x.to_numpy()), index=x.index)
    wallet = backtest(frames, predictions, report["thresholds"], frozen.risk, frozen.market.timeframe, start, end)
    result = metrics(wallet)
    if not np.isclose(result["final_equity"], report["test"]["final_equity"], rtol=1e-9):
        raise ValueError("Reproduction differs from the saved test; inspect data/model versions")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Fly Trade — virtual-money crypto research and paper trading")
    result.add_argument("--config", default="config.toml", help="Path to config.toml (put before command)")
    sub = result.add_subparsers(dest="command")
    for name, help_text in [("run", "Download, train, evaluate, then start/resume paper trading"),
                            ("paper", "Start/resume the saved forward paper experiment")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--once", action="store_true", help="Process one live market snapshot and exit")
        cmd.add_argument("--no-dashboard", action="store_true")
    sub.add_parser("download", help="Download/resume real historical candles")
    sub.add_parser("train", help="Download history, train, validate, test, and prepare a paper model")
    cmd = sub.add_parser("backtest", help="Reproduce the frozen held-out test from saved data/models")
    cmd.add_argument("--run-id")
    sub.add_parser("brain-download", help="Download and prepare actual FlyWire connectivity")
    sub.add_parser("status", help="Print experiment and wallet status")
    sub.add_parser("report", help="Print the saved research report and its location")
    sub.add_parser("dashboard", help="View existing research/paper results in the local dashboard")
    cmd = sub.add_parser("demo", help="Offline SYNTHETIC software demo; saved separately under data/demo")
    cmd.add_argument("--bars", type=int, default=4000)
    cmd.add_argument("--serve", action="store_true", help="Keep a dashboard open after the demo")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command or "run"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    server, config = None, None
    stop = Event()
    try:
        config = load_config(args.config)
        if command == "demo":
            config = replace(config, app=replace(config.app, data_dir=str(config.data_dir / "demo")))
            config = replace(config, model=replace(config.model, kind="mushroom"))
        if command == "status":
            state = snapshot(config)
            if state["paper"]:
                print(json.dumps({k: state["paper"][k] for k in ("status", "run_id", "last_poll", "last_error", "metrics")}, indent=2))
            else:
                print(json.dumps({"status": "No forward paper wallet", "report": state["report"]["test"] if state["report"] else None}, indent=2))
            return
        if command == "report":
            report = load_report(config)
            print(json.dumps(report, indent=2))
            print(config.data_dir / "runs" / report["run_id"] / "report.html")
            return
        if command == "dashboard":
            server = start_dashboard(config)
            stop.wait()
            return
        with ProcessLock(config.data_dir / "process.lock"):
            if command in ("run", "paper") and not getattr(args, "no_dashboard", False):
                try:
                    server = start_dashboard(config)
                except OSError as exc:
                    LOG.warning("Dashboard unavailable (%s); trading still runs. Change app.dashboard_port if needed.", exc)
            if command == "brain-download":
                print(prepare_graph(config))
                return
            if command == "backtest":
                print(json.dumps(reproduce_test(config, args.run_id), indent=2))
                return
            if command == "demo":
                if not 1000 <= args.bars <= 100000:
                    raise ValueError("Demo bars must be between 1000 and 100000")
                LOG.info("SYNTHETIC OFFLINE DEMO — these prices and returns are fabricated test inputs")
                report = run_research(config, synthetic_history(config, args.bars), synthetic=True)
                print(json.dumps({"synthetic": True, "test": report["test"], "report": str(config.data_dir / "runs" / report["run_id"] / "report.html")}, indent=2))
                if args.serve:
                    server = start_dashboard(config)
                    stop.wait()
                return
            needs_training = command == "train"
            if command == "run" and not (config.data_dir / "paper.json").exists():
                try:
                    existing = load_report(config)
                    needs_training = existing["fingerprint"] != config.trading_fingerprint() or existing["synthetic"]
                except (FileNotFoundError, ValueError):
                    needs_training = True
            if command == "download" or needs_training:
                atomic_json(config.data_dir / "progress.json", {"message": "Downloading historical candles", "error": None})
                LOG.info("Downloading %d days of historical crypto candles; first run needs internet access", config.market.history_days)
                frames = download_history(config)
                if command == "download":
                    print(json.dumps({s: len(f) for s, f in frames.items()}, indent=2))
                    return
                atomic_json(config.data_dir / "progress.json", {"message": "Training and evaluating", "error": None})
                report = run_research(config, frames)
                atomic_json(config.data_dir / "progress.json", {"message": "Research complete", "error": None})
                if command == "train":
                    print(json.dumps(report["test"], indent=2))
                    print(config.data_dir / "runs" / report["run_id"] / "report.html")
                    return
            LOG.info("Virtual-money mode only. No account, API key, or deposit is used. Ctrl+C stops the process.")
            runner = PaperRunner(config)
            # SIGTERM (e.g. systemd) saves state at the current tick boundary.
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            runner.run(stop, once=getattr(args, "once", False))
    except KeyboardInterrupt:
        LOG.info("Stopped. Paper wallet and consumed signals remain saved.")
    except (ValueError, RuntimeError, OSError) as exc:
        LOG.error("%s", exc)
        if config is not None:
            atomic_json(config.data_dir / "progress.json", {"message": "Stopped", "error": str(exc)})
        sys.exit(1)
    finally:
        if server:
            server.shutdown()
            server.server_close()
