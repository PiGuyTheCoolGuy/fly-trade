"""Strict, versionable configuration; no credential or real-money settings."""

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib


@dataclass
class Market:
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    timeframe: int = 300
    history_days: int = 90
    poll_seconds: float = 15
    request_interval: float = 0.35
    timeout_seconds: float = 20


@dataclass
class Model:
    kind: str = "mushroom"
    seed: int = 42
    hidden_size: int = 256
    active_fraction: float = 0.1
    ridge: float = 10.0
    horizon: int = 12
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    brain_neurons: int = 2048
    brain_steps: int = 3


@dataclass
class Risk:
    initial_cash: float = 10000
    max_position_fraction: float = 0.2
    max_total_exposure: float = 0.6
    risk_per_trade: float = 0.005
    stop_loss: float = 0.02
    take_profit: float = 0.04
    max_hold_bars: int = 72
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.1
    fee_bps: float = 60
    slippage_bps: float = 5
    spread_bps: float = 10
    min_order: float = 10


@dataclass
class App:
    data_dir: str = "data"
    dashboard_port: int = 8787
    quote_max_age_seconds: int = 60


@dataclass
class Config:
    market: Market = field(default_factory=Market)
    model: Model = field(default_factory=Model)
    risk: Risk = field(default_factory=Risk)
    app: App = field(default_factory=App)
    root: Path = field(default_factory=Path.cwd)

    @property
    def data_dir(self) -> Path:
        return (self.root / self.app.data_dir).resolve()

    def snapshot(self) -> dict:
        return {k: asdict(getattr(self, k)) for k in ("market", "model", "risk", "app")}

    def trading_fingerprint(self) -> str:
        # Cosmetic/network settings may change while a wallet is running.
        value = {"symbols": self.market.symbols, "timeframe": self.market.timeframe,
                 "risk": asdict(self.risk), "model": asdict(self.model)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def validate(self) -> None:
        m, b, r, a = self.market, self.model, self.risk, self.app
        for section in (m, b, r, a):
            for key, value in asdict(section).items():
                if isinstance(value, (float, int)) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
        if (not isinstance(m.symbols, list) or not m.symbols
                or len(set(m.symbols)) != len(m.symbols)
                or any(not re.fullmatch(r"[A-Z0-9]+-USD", s) for s in m.symbols)):
            raise ValueError("Use unique Coinbase USD spot symbols, e.g. BTC-USD")
        for section, keys in ((m, ["timeframe", "history_days"]),
                              (b, ["seed", "hidden_size", "horizon", "brain_neurons", "brain_steps"]),
                              (r, ["max_hold_bars"]), (a, ["dashboard_port", "quote_max_age_seconds"])):
            for key in keys:
                if type(getattr(section, key)) is not int:
                    raise ValueError(f"{key} must be an integer")
        if m.timeframe not in (60, 300, 900, 3600, 21600, 86400):
            raise ValueError("Unsupported Coinbase candle timeframe")
        if min(m.history_days, m.poll_seconds, m.timeout_seconds) <= 0 or m.request_interval < 0:
            raise ValueError("Market durations must be positive")
        if b.kind not in ("mushroom", "flywire"):
            raise ValueError("model.kind must be mushroom or flywire")
        if b.hidden_size < 8 or b.horizon < 1 or b.ridge <= 0 or b.brain_neurons < 0 or b.brain_steps < 1:
            raise ValueError("Invalid model size, horizon, ridge, or brain settings")
        if not 0 < b.active_fraction <= 1 or b.seed < 0:
            raise ValueError("Invalid model sparsity or seed")
        if not (0.1 <= b.train_fraction < 0.9 and 0.05 <= b.validation_fraction
                and b.train_fraction + b.validation_fraction <= 0.9):
            raise ValueError("Train/validation fractions must leave at least 10% for testing")
        for key in ("max_position_fraction", "max_total_exposure", "risk_per_trade",
                    "stop_loss", "take_profit", "daily_loss_limit", "max_drawdown"):
            if not 0 < getattr(r, key) <= 1:
                raise ValueError(f"risk.{key} must be in (0, 1]")
        if min(r.initial_cash, r.min_order, r.max_hold_bars) <= 0:
            raise ValueError("Cash, minimum order, and holding limit must be positive")
        if any(not 0 <= getattr(r, k) < 1000 for k in ("fee_bps", "spread_bps", "slippage_bps")):
            raise ValueError("Costs must be between 0 and 1000 basis points")
        if not 1 <= a.dashboard_port <= 65535 or a.quote_max_age_seconds <= 0:
            raise ValueError("Invalid dashboard port or quote age")


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    classes = {"market": Market, "model": Model, "risk": Risk, "app": App}
    if set(raw) - classes.keys():
        raise ValueError(f"Unknown config sections: {set(raw) - classes.keys()}")
    config = Config(root=path.parent)
    for name, cls in classes.items():
        values = raw.get(name, {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {unknown}")
        setattr(config, name, cls(**values))
    config.validate()
    return config
