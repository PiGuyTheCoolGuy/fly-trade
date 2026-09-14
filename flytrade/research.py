"""Chronological training, purged labels, validation selection and untouched test."""

from datetime import datetime, timezone
import hashlib
import html
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .config import Config
from .connectome import graph_path, prepare_graph
from .engine import backtest, buy_and_hold, metrics
from .features import WARMUP, features, targets
from .market import iso, validate_frame
from .model import FlyModel
from .storage import atomic_json

LOG = logging.getLogger(__name__)


def boundaries(n: int, config: Config) -> tuple[int, int, int]:
    train_end = int(n * config.model.train_fraction)
    test_start = int(n * (config.model.train_fraction + config.model.validation_fraction))
    fit_end = train_end - config.model.horizon - 1
    if fit_end - WARMUP < 100 or min(test_start - train_end, n - test_start) < 50:
        raise ValueError("Too few candles for purged train/validation/test splits; increase history_days")
    return fit_end, train_end, test_start


def run_research(config: Config, frames: dict, *, synthetic: bool = False) -> dict:
    tf = config.market.timeframe
    for df in frames.values():
        validate_frame(df, tf)
    timeline = next(iter(frames.values())).index
    if any(not f.index.equals(timeline) for f in frames.values()):
        raise ValueError("Market histories must share one complete timeline")
    if set(frames) != set(config.market.symbols):
        raise ValueError("History symbols do not match config")
    n = len(timeline)
    fit_end, train_end, test_start = boundaries(n, config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = config.data_dir / "runs" / run_id
    directory.mkdir(parents=True)
    graph = None
    if config.model.kind == "flywire":
        graph = graph_path(config)
        if not graph.exists():
            graph = prepare_graph(config)
    trained, feature_frames, all_predictions = {}, {}, {}
    for symbol, df in frames.items():
        LOG.info("Training %s (%s)", symbol, config.model.kind)
        x, y = features(df), targets(df, config.model.horizon)
        model = FlyModel(config.model, x.shape[1], graph)
        model.fit(x.iloc[WARMUP:fit_end].to_numpy(), y.iloc[WARMUP:fit_end].to_numpy())
        model.metadata.update({"symbol": symbol, "timeframe": tf, "feature_names": list(x.columns),
                               "training_label_end": int(timeline[train_end - 1]), "run_id": run_id})
        model.save(directory / f"{symbol}-evaluation.npz")
        predicted = model.predict(x.iloc[train_end:].to_numpy())
        all_predictions[symbol] = pd.Series(predicted, index=timeline[train_end:])
        trained[symbol], feature_frames[symbol] = model, x
    # Threshold selection sees only the validation portfolio. Infinity is a
    # legitimate stay-in-cash candidate, represented as 1.0 in saved JSON.
    one_way = (config.risk.slippage_bps + config.risk.spread_bps / 2) / 10000
    fee = config.risk.fee_bps / 10000
    cost_hurdle = float(np.log((1 + one_way) * (1 + fee) / ((1 - one_way) * (1 - fee))))
    hurdles = sorted(set([1.0, max(cost_hurdle, 0.001), max(cost_hurdle * 1.5, 0.002),
                           max(cost_hurdle * 2, 0.004), max(cost_hurdle * 3, 0.008)]), reverse=True)
    candidates = []
    best_score, chosen = -float("inf"), 1.0
    validation_predictions = {s: p[p.index < timeline[test_start]] for s, p in all_predictions.items()}
    for threshold in hurdles:
        thresholds = {s: threshold for s in frames}
        wallet = backtest(frames, validation_predictions, thresholds, config.risk, tf,
                          int(timeline[train_end]), int(timeline[test_start]))
        stats = metrics(wallet)
        candidates.append({"threshold": threshold, **stats})
        if stats["final_equity"] > best_score + 1e-8:
            best_score, chosen = stats["final_equity"], threshold
    thresholds = {s: chosen for s in frames}
    test_predictions = {s: p[p.index >= timeline[test_start]] for s, p in all_predictions.items()}
    start, end = int(timeline[test_start]), int(timeline[-1]) + tf
    wallet = backtest(frames, test_predictions, thresholds, config.risk, tf, start, end)
    ema_predictions = {}
    for symbol, frame in frames.items():
        fast = frame.close.ewm(span=12, adjust=False).mean()
        slow = frame.close.ewm(span=48, adjust=False).mean()
        ema_predictions[symbol] = pd.Series(np.where(fast > slow, 1.0, -1.0), index=frame.index).loc[start:]
    ema_wallet = backtest(frames, ema_predictions, {s: 0 for s in frames}, config.risk, tf, start, end)
    checksums = {}
    for symbol, frame in frames.items():
        # Refit only after the held-out test has been frozen. The deployment
        # artifact is separate and never used to report that earlier test.
        x, y = feature_frames[symbol], targets(frame, config.model.horizon)
        deploy_end = n - config.model.horizon - 1
        deploy = FlyModel(config.model, x.shape[1], graph)
        deploy.fit(x.iloc[WARMUP:deploy_end].to_numpy(), y.iloc[WARMUP:deploy_end].to_numpy())
        deploy.metadata.update({"symbol": symbol, "timeframe": tf, "feature_names": list(x.columns),
                                "training_label_end": int(timeline[-1]), "run_id": run_id})
        model_file = directory / f"{symbol}-paper.npz"
        deploy.save(model_file)
        checksums[symbol] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    stats = metrics(wallet)
    summary = {
        "schema_version": 1, "run_id": run_id, "synthetic": synthetic,
        "created_at": iso(datetime.now(timezone.utc).timestamp()),
        "model_kind": config.model.kind,
        "model_description": "Fly-inspired sparse expansion (synthetic wiring)" if config.model.kind == "mushroom"
                             else "Real FlyWire graph, simplified tanh reservoir; learned readout",
        "symbols": list(frames), "candles_per_symbol": n, "config": config.snapshot(),
        "fingerprint": config.trading_fingerprint(), "timeframe": tf,
        "ranges": {"history_start": iso(int(timeline[0])), "history_end_exclusive": iso(end),
                   "train_end_exclusive": iso(int(timeline[train_end])),
                   "validation_end_exclusive": iso(start), "test_start": iso(start),
                   "test_end_exclusive": iso(end), "purged_training_signals": config.model.horizon + 1},
        "thresholds": thresholds, "cost_hurdle": cost_hurdle, "validation_candidates": candidates,
        "test": stats, "buy_and_hold": buy_and_hold(frames, config.risk, start, end),
        "ema_baseline": metrics(ema_wallet), "cash_baseline": {"return_pct": 0.0},
        "paper_model_sha256": checksums, "paper_training_end": int(timeline[-1]),
        "evaluation_reused_for_paper": False,
        "history_sha256": {s: hashlib.sha256(df.to_csv().encode()).hexdigest() for s, df in frames.items()},
        "equity_curve": wallet.curve[::max(1, len(wallet.curve) // 600)] + [wallet.curve[-1]],
        "notes": ["Synthetic data is for software testing only." if synthetic else "Real historical candles; virtual fills.",
                  "Test results are not evidence of future profitability.",
                  "The deployed model is refit on matured labels after the test; judge it using forward paper results.",
                  "Backtests assume fixed spread/slippage; no order-book depth, latency, queue position or partial fills.",
                  "Stay-in-cash can win validation. No trades is a valid outcome."],
    }
    if graph:
        summary["connectome"] = json.loads(graph.with_suffix(".json").read_text())
    for symbol, frame in frames.items():
        frame.to_csv(directory / f"{symbol}-candles.csv.gz", compression="gzip")
    pd.DataFrame(wallet.fills, columns=["timestamp", "symbol", "side", "price", "quantity", "fee", "reason"]).to_csv(directory / "fills.csv", index=False)
    pd.DataFrame(wallet.closed).to_csv(directory / "closed_trades.csv", index=False)
    pd.DataFrame(wallet.curve).to_csv(directory / "equity.csv", index=False)
    atomic_json(directory / "report.json", summary)
    (directory / "report.html").write_text(report_html(summary), encoding="utf-8")
    atomic_json(config.data_dir / "latest.json", {"run_id": run_id})
    LOG.info("Held-out return %.2f%%, max drawdown %.2f%%, %d closed trades",
             stats["return_pct"], stats["max_drawdown_pct"], stats["closed_trades"])
    if chosen == 1.0:
        LOG.info("Validation selected cash: this model has not demonstrated an edge after costs")
    return summary


def report_html(report: dict) -> str:
    esc = html.escape
    rows = "".join(f"<tr><td>{esc(k.replace('_', ' '))}</td><td>{esc(str(v))}</td></tr>" for k, v in report["test"].items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Fly Trade research report</title>
    <style>body{{font:16px system-ui;background:#0c1420;color:#e7eef7;max-width:900px;margin:50px auto;padding:24px}}
    h1{{color:#7befc1}}table{{width:100%;border-collapse:collapse}}td{{padding:12px;border-bottom:1px solid #324052}}
    a{{color:#7befc1}}small{{color:#b1becf}}</style><h1>Fly Trade · {'SYNTHETIC DEMO' if report['synthetic'] else 'Historical test'}</h1>
    <p>{esc(report['model_description'])}</p><p>Held-out period: {esc(report['ranges']['test_start'])} to {esc(report['ranges']['test_end_exclusive'])}</p>
    <table>{rows}</table><p>Buy and hold: {report['buy_and_hold']['return_pct']:.2f}% · EMA baseline: {report['ema_baseline']['return_pct']:.2f}% · Cash: 0%</p>
    <p>{'</p><p>'.join(esc(x) for x in report['notes'])}</p>
    <p><a href="report.json">Full settings and evidence</a> · <a href="fills.csv">Fills</a> · <a href="equity.csv">Equity curve</a></p>
    <small>Run {esc(report['run_id'])}; fee assumption {report['config']['risk']['fee_bps']} bps per side.</small></html>"""


def load_report(config: Config, run_id: str | None = None) -> dict:
    if run_id is None:
        latest = config.data_dir / "latest.json"
        if not latest.exists():
            raise ValueError("No trained model; run: python run.py train")
        run_id = json.loads(latest.read_text())["run_id"]
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")
    return json.loads((config.data_dir / "runs" / run_id / "report.json").read_text())
