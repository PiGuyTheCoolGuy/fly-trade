"""Cash/position accounting shared by historical and forward simulation."""

from dataclasses import asdict, dataclass, field
import math

from .config import Risk


@dataclass
class Position:
    quantity: float
    entry_price: float
    entry_fee: float
    opened_at: int
    stop: float
    take: float


@dataclass
class Wallet:
    cash: float
    initial_cash: float
    peak: float
    day_start: float
    day: int = -1
    daily_halted: bool = False
    halted: bool = False
    positions: dict[str, Position] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    fills: list[dict] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, cash: float):
        return cls(cash, cash, cash, cash)

    @property
    def equity(self) -> float:
        return self.cash + self.exposure

    @property
    def exposure(self) -> float:
        return sum(p.quantity * self.marks.get(s, p.entry_price) for s, p in self.positions.items())

    def begin_day(self, timestamp: int) -> None:
        day = timestamp // 86400
        if day != self.day:
            self.day = day
            self.day_start = self.equity
            self.daily_halted = False

    def risk_check(self, risk: Risk) -> bool:
        self.peak = max(self.peak, self.equity)
        self.daily_halted |= self.equity <= self.day_start * (1 - risk.daily_loss_limit)
        self.halted |= self.equity <= self.peak * (1 - risk.max_drawdown)
        return self.halted or self.daily_halted

    def buy(self, symbol: str, price: float, timestamp: int, risk: Risk, reason="model") -> bool:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid fill price")
        if symbol in self.positions or self.risk_check(risk):
            return False
        fee_rate = risk.fee_bps / 10000
        # Include a cost buffer in the stop-based risk budget.
        unit_risk = price * (risk.stop_loss + 2 * fee_rate
                             + (2 * risk.slippage_bps + risk.spread_bps) / 10000)
        available_exposure = max(0, self.equity * risk.max_total_exposure - self.exposure)
        notional = min(self.equity * risk.max_position_fraction, available_exposure,
                       self.cash / (1 + fee_rate), self.equity * risk.risk_per_trade / unit_risk * price)
        if notional < risk.min_order:
            return False
        quantity = notional / price
        fee = notional * fee_rate
        self.cash = max(0.0, self.cash - notional - fee)
        self.positions[symbol] = Position(quantity, price, fee, timestamp,
                                          price * (1 - risk.stop_loss), price * (1 + risk.take_profit))
        self.fills.append({"timestamp": timestamp, "symbol": symbol, "side": "BUY",
                           "price": price, "quantity": quantity, "fee": fee, "reason": reason})
        return True

    def sell(self, symbol: str, price: float, timestamp: int, risk: Risk, reason="model") -> bool:
        if symbol not in self.positions:
            return False
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid fill price")
        pos = self.positions.pop(symbol)
        proceeds = pos.quantity * price
        fee = proceeds * risk.fee_bps / 10000
        self.cash += proceeds - fee
        pnl = proceeds - fee - pos.quantity * pos.entry_price - pos.entry_fee
        self.fills.append({"timestamp": timestamp, "symbol": symbol, "side": "SELL",
                           "price": price, "quantity": pos.quantity, "fee": fee, "reason": reason})
        self.closed.append({"symbol": symbol, "opened_at": pos.opened_at, "closed_at": timestamp,
                            "entry_price": pos.entry_price, "exit_price": price,
                            "quantity": pos.quantity, "pnl": pnl, "reason": reason})
        return True

    def record(self, timestamp: int) -> None:
        self.peak = max(self.peak, self.equity)
        point = {"timestamp": timestamp, "equity": self.equity, "cash": self.cash,
                 "exposure": self.exposure, "drawdown": 1 - self.equity / self.peak}
        if self.curve and self.curve[-1]["timestamp"] == timestamp:
            self.curve[-1] = point
        else:
            self.curve.append(point)

    def serialize(self) -> dict:
        return asdict(self)

    @classmethod
    def deserialize(cls, value: dict):
        value = dict(value)
        value["positions"] = {s: Position(**p) for s, p in value["positions"].items()}
        return cls(**value)


def historical_fill(price: float, side: str, risk: Risk) -> float:
    cost = (risk.slippage_bps + risk.spread_bps / 2) / 10000
    return price * (1 + cost if side == "BUY" else 1 - cost)


def metrics(wallet: Wallet) -> dict:
    winners = [t["pnl"] for t in wallet.closed if t["pnl"] > 0]
    losers = [t["pnl"] for t in wallet.closed if t["pnl"] < 0]
    return {"initial_equity": wallet.initial_cash, "final_equity": wallet.equity,
            "return_pct": (wallet.equity / wallet.initial_cash - 1) * 100,
            "max_drawdown_pct": max([p["drawdown"] for p in wallet.curve] or [0]) * 100,
            "closed_trades": len(wallet.closed), "fills": len(wallet.fills),
            "win_rate_pct": 100 * len(winners) / len(wallet.closed) if wallet.closed else None,
            "profit_factor": sum(winners) / -sum(losers) if losers else None,
            "fees_paid": sum(t["fee"] for t in wallet.fills),
            "realized_pnl": sum(t["pnl"] for t in wallet.closed), "risk_halted": wallet.halted}


def backtest(frames: dict, predictions: dict, thresholds: dict, risk: Risk,
             timeframe: int, start: int, end: int) -> Wallet:
    """Signals from close[t-1], fills at open[t]; conservative same-bar stops."""
    wallet = Wallet.new(risk.initial_cash)
    timeline = next(iter(frames.values())).index
    times = timeline[(timeline >= start) & (timeline < end)]
    if len(times) < 2:
        raise ValueError("Backtest requires at least two candles")
    for frame in frames.values():
        if not frame.index.equals(timeline):
            raise ValueError("All symbols must have the same complete candle timeline")
    wallet.record(int(times[0]) - 1)
    for ts in times:
        ts = int(ts)
        wallet.begin_day(ts)
        bars = {s: df.loc[ts] for s, df in frames.items()}
        wallet.marks.update({s: float(bar.open) for s, bar in bars.items()})
        halted = wallet.risk_check(risk)
        exited = set()
        for symbol, pos in list(wallet.positions.items()):
            bar = bars[symbol]
            pred = predictions[symbol].get(ts - timeframe, float("nan"))
            reason = None
            if halted:
                reason = "risk_limit"
            elif bar.open <= pos.stop:
                reason = "stop_gap"
            elif bar.open >= pos.take:
                reason = "take_gap"
            elif ts - pos.opened_at >= risk.max_hold_bars * timeframe:
                reason = "max_hold"
            elif math.isfinite(pred) and pred <= 0:
                reason = "model_exit"
            if reason:
                wallet.sell(symbol, historical_fill(float(bar.open), "SELL", risk), ts, risk, reason)
                exited.add(symbol)
        if not wallet.risk_check(risk):
            # Configuration order provides deterministic allocation priority.
            for symbol, bar in bars.items():
                pred = predictions[symbol].get(ts - timeframe, float("nan"))
                if symbol not in exited and math.isfinite(pred) and pred > thresholds[symbol]:
                    wallet.buy(symbol, historical_fill(float(bar.open), "BUY", risk), ts, risk)
        for symbol, pos in list(wallet.positions.items()):
            bar = bars[symbol]
            # OHLC does not tell us event order: stop wins when both are touched.
            trigger = None
            if bar.low <= pos.stop:
                trigger = (min(float(bar.open), pos.stop), "stop_loss")
            elif bar.high >= pos.take:
                trigger = (pos.take, "take_profit")
            if trigger:
                wallet.sell(symbol, historical_fill(trigger[0], "SELL", risk), ts, risk, trigger[1])
        wallet.marks.update({s: float(bar.close) for s, bar in bars.items()})
        if wallet.risk_check(risk):
            for symbol in list(wallet.positions):
                wallet.sell(symbol, historical_fill(float(bars[symbol].close), "SELL", risk),
                            ts + timeframe - 1, risk, "risk_limit")
        wallet.record(ts + timeframe - 1)
    # Liquidate the last mark with costs so an open winner cannot inflate results.
    last = int(times[-1]) + timeframe - 1
    for symbol in list(wallet.positions):
        wallet.sell(symbol, historical_fill(wallet.marks[symbol], "SELL", risk), last, risk, "end_of_test")
    wallet.record(last)
    return wallet


def buy_and_hold(frames: dict, risk: Risk, start: int, end: int) -> dict:
    # Fully invested equal-weight benchmark, including entry and exit costs.
    amounts = []
    portion = risk.initial_cash / len(frames)
    for frame in frames.values():
        period = frame[(frame.index >= start) & (frame.index < end)]
        first = historical_fill(float(period.iloc[0].open), "BUY", risk)
        last = historical_fill(float(period.iloc[-1].close), "SELL", risk)
        quantity = portion / (first * (1 + risk.fee_bps / 10000))
        amounts.append(quantity * last * (1 - risk.fee_bps / 10000))
    equity = sum(amounts)
    return {"final_equity": equity, "return_pct": (equity / risk.initial_cash - 1) * 100,
            "description": "100% invested, equal weights, buy first open/sell last close; costs included"}
