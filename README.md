# Fly Trade

An experimental, CPU-friendly crypto trading program with **simulated money**.
Start it once: it downloads historical candles, trains a model, evaluates a held-out
period, then watches real market prices and makes virtual trades. A local dashboard
shows the wallet, signals, fills, and test results.

**There is no real-money trading implementation, API-key field, or exchange order
endpoint in this version.** It needs internet access to market data, not an exchange
account. Future live execution will be a separate development step.

## Start on your Ubuntu server

Install **Python 3.10 or newer**, Git, and Python's venv support. Check with
`python3 --version`. Ubuntu 22.04's standard Python 3.10 is supported.

```bash
git clone https://github.com/PiGuyTheCoolGuy/fly-trade.git
cd fly-trade
bash start.sh
```

The launcher creates `.venv` and installs dependencies on first use. It then runs
the complete workflow. Open **http://127.0.0.1:8787** on that computer.

If running on `treicserver`, forward the dashboard to your laptop:

```bash
ssh -L 8787:127.0.0.1:8787 treic@YOUR_SERVER_IP
```

Then open http://127.0.0.1:8787 on your laptop. The dashboard listens only on the
server's loopback interface; it is a local, read-only monitor.

Press **Ctrl+C** to stop. Run the same command again to resume the saved wallet.
Keep the process running for forward simulation. When it is stopped, positions
remain open in the virtual wallet and stop orders are not monitored.

## Start on Windows

Install Python 3.10+ and Git. In PowerShell:

```powershell
git clone https://github.com/PiGuyTheCoolGuy/fly-trade.git
cd fly-trade
.\start.bat
```

Open http://127.0.0.1:8787. You can also double-click `start.bat` after downloading
and extracting the repository ZIP.

## Default experiment

| Setting | Default |
|---|---|
| Data source | Coinbase Exchange public spot market data |
| Markets | BTC-USD, ETH-USD, SOL-USD |
| Candle interval | 5 minutes |
| Historical period | Previous 90 days; automatically paginated and cached |
| Starting wallet | $10,000 shared across the three markets |
| Position direction | Long or cash; no shorts, borrowing, or leverage |
| Prediction horizon | 12 bars, or one hour |
| Max holding time | 72 bars, or six hours |
| Position cap | 20% of current equity per market |
| Total position cap | 60% of current equity |
| Planned risk per trade | 0.5% of equity, including a cost buffer |
| Stop / take profit | 2% / 4% from simulated entry price |
| Daily loss limit | 3%; close positions and pause entries until the next UTC day |
| Drawdown limit | 10% from portfolio peak; close positions and latch a halt |
| Fee assumption | 60 basis points (0.60%) **per side** |
| Backtest spread / slippage | 10 bps full spread / 5 bps adverse slippage per side |

All settings live in [`config.toml`](config.toml). The fee setting is an editable
assumption, **not a claim about your Coinbase fee tier**. Forward paper fills use
the observed bid/ask spread plus the slippage assumption and fees.

Signals are calculated from completed candles. Price/stop checks run every 15
seconds by default. Orders fill only in the virtual wallet. The specified risk
budget is not a guaranteed maximum loss: gaps, downtime, and price jumps can exceed it.

The first run spends time downloading data before training. Progress appears in
the terminal and dashboard. Repeated runs reuse complete cached pages. A persistent
gap aborts the download with a clear error; prices are never filled in to hide gaps.
The three defaults are an initial test universe, not investment recommendations.

## What the “fly brain” means here

This repo provides two clearly labeled experimental models:

| Mode | What it uses | What learns |
|---|---|---|
| `mushroom` (default) | A small, randomly wired sparse expansion inspired by insect mushroom bodies | A ridge-regression output layer predicting a future log return |
| `flywire` | Actual signed, directed FlyWire v783 connectivity from the Shiu research repository, transformed into a bounded graph reservoir | A ridge-regression output layer; biological connections remain fixed |

**The default is not an emulated fly brain.** It is a fast baseline you can run on
a CPU, including older servers. The FlyWire option uses genuine connectivity but
applies simplified `tanh` dynamics, artificial market inputs, and pooled outputs.
It does not reproduce Shiu's biophysical leaky integrate-and-fire model, simulate
a conscious fly, or establish that fly wiring is useful for trading.

The market inputs include causal returns, volatility, trend, candle range/body,
volume, RSI, and time of day. For the default model, sparse expansion and
winner-take-all activation feed a learned readout. Training is supervised, not
reinforcement learning or an LLM agent. The model is frozen during each forward
paper experiment so its results stay interpretable.

### Use actual FlyWire connectivity

After the normal setup, install the optional Parquet reader:

```bash
.venv/bin/python -m pip install -e ".[connectome]"
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Make a copy of `config.toml`, for example `flywire.toml`. In that copy set:

```toml
[model]
kind = "flywire"
brain_neurons = 2048
# Keep or customize the other model settings from config.toml.

[app]
data_dir = "data-flywire"
dashboard_port = 8787
quote_max_age_seconds = 60
```

These are changes to the existing sections, not additional duplicate sections.
Then run:

```bash
.venv/bin/python run.py --config flywire.toml brain-download
.venv/bin/python run.py --config flywire.toml
```

`brain-download` fetches approximately 104 MB of pinned upstream data and verifies
its Git blob checksums. It defaults to an induced subgraph of the 2,048 neurons with
the largest weighted degree, retaining only connections between selected neurons.
That subset is a computational experiment, not a biological functional circuit.
Set `brain_neurons = 0` to use every neuron in the upstream list; this is substantially
more expensive. No GPU is required. Allow extra RAM for Parquet parsing and sparse
graph construction. The full-graph runtime has not been benchmarked on your server.

The selected graph, original neuron IDs, source revision, transformations, node/edge
counts, and SHA-256 hash are saved under the experiment's `brain/` directory.
No upstream simulation code is executed or vendored. Attribution and source links
are in [`docs/SOURCES.md`](docs/SOURCES.md).

## How to judge results

1. **Training:** the first 60% of candles fit both normalization and the model.
2. **Validation:** the next 20% selects an entry threshold after simulated costs.
   A stay-in-cash candidate participates in selection.
3. **Held-out test:** the last 20% is evaluated once with the selected rule.
   It is compared with cash, an EMA trend strategy under the same risk limits, and
   a fully invested equal-weight buy-and-hold benchmark with costs.
4. **Forward paper trading:** after saving the test result, a *separate* deployment
   model is fit on all labels that have matured. It observes new real market quotes.
   Its performance must be measured forward; the earlier test is not its track record.

Training labels are purged at split boundaries. A signal observed at candle `t` can
only fill at `t+1`'s opening price in a backtest. Historical stops assume the loss
occurs first when a candle touches both stop and take-profit levels. All remaining
test positions are liquidated with costs at the end. Reports include exact date
ranges, configuration, data hashes, validation choices, and the selected model.

**Zero trades is a valid result.** If validation selects cash, the dashboard displays
“Cash selected.” It means this test did not find enough benefit after costs; the
program does not force trades to make the display look active. Lower fees should
only represent a realistic assumption, not a way to manufacture a profitable test.

Review forward results over different market conditions, sufficient independent
trades, drawdowns, fee sensitivity, and comparisons with simple baselines. Repeatedly
tuning to the held-out result effectively turns it into training data; use a fresh
future period for the next assessment. Profitable simulated returns do not establish
that real-money trading will be profitable.

## Commands

After the launcher has installed dependencies, activate your environment:

```bash
source .venv/bin/activate
# PowerShell alternative: .\.venv\Scripts\Activate.ps1
```

| Command | Purpose |
|---|---|
| `python run.py` | Complete setup and forward paper trading; resume existing experiment |
| `python run.py download` | Download/resume the requested historical window |
| `python run.py train` | Train, validate, test, and save a separate deployment model |
| `python run.py backtest` | Reproduce the frozen test with saved data and evaluation model |
| `python run.py paper` | Start/resume forward paper trading with the saved model |
| `python run.py paper --once --no-dashboard` | Process one fresh market snapshot and exit |
| `python run.py dashboard` | View existing results without starting trades |
| `python run.py status` | Print virtual wallet status |
| `python run.py report` | Print research details and the HTML report location |
| `python run.py demo --serve` | Offline synthetic software test and dashboard |
| `python run.py brain-download` | Fetch and prepare the actual FlyWire graph |

The optional `--config PATH` argument goes **before** the command. Launchers pass
arguments through, e.g. `bash start.sh demo --serve` or `.\start.bat demo --serve`.
The demo uses generated prices and saves separately in `data/demo/`; it cannot
silently replace your normal wallet or start forward trading with synthetic training.

For a manual install with a particular interpreter:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python run.py
```

## Saved files and new experiments

The configured data directory contains:

- `candles.sqlite3`: reusable, deduplicated historical candles.
- `paper.json`: atomic wallet checkpoint, consumed signals, complete fill ledger,
  closed trades, equity samples, current status, and model-run identity.
- `runs/<run-id>/report.html` and `report.json`: human-readable and detailed reports.
- `runs/<run-id>/fills.csv`, `closed_trades.csv`, `equity.csv`: historical test exports.
- `runs/<run-id>/*-candles.csv.gz`: the exact historical snapshots used by that run.
- Separate `*-evaluation.npz` and `*-paper.npz` model files, loaded without pickle.
- `latest.json`: the newest completed research run. An existing wallet stays pinned
  to its original model even if you train another one.

Back up the entire data directory. Git ignores it. Keep it together when moving to
another computer. Wallet state and consumed signal IDs save together, and an OS
process lock prevents two trading processes from using the same directory. After a
network outage, only the newest completed signal can trade at a current quote;
missing historical signals are not replayed as fictional live fills.

To change markets, risk limits, model settings, or start fresh after a drawdown halt,
copy the config and select a **new `app.data_dir`**. Then launch that config. This
preserves the old experiment and prevents an accidental balance/model reset. A
drawdown halt does not automatically unlock when you restart.

## Run as a service or in Docker

An editable systemd example is in [`scripts/flytrade.service`](scripts/flytrade.service).
Complete one successful normal setup first. Replace `YOUR_USER` and both paths,
copy the file into `/etc/systemd/system/`, then use `systemctl daemon-reload` and
`systemctl enable --now flytrade`. View logs with `journalctl -u flytrade -f`.

Alternatively:

```bash
docker compose up --build -d
docker compose logs -f
```

The container is headless; `data/` is mounted persistently. Use the HTML/JSON/CSV
reports or `docker compose exec flytrade python run.py status`. The supplied
container doesn't expose a dashboard port. Stop it with `docker compose down`.

## Verification

```bash
python -m unittest discover -s tests -v
python run.py demo --bars 1600
```

Tests cover cash accounting, fees on both sides, position caps, no shorts, causal
features, label purging, next-open fills, stop/take ambiguity, price gaps, holding
limits, risk halts, download pagination/resumption, rate-limit retries, invalid
data, stale/auction quotes, exact model round trips, separate test/deployment
artifacts, paper restart idempotency, outages, config/model mismatch protection,
and dashboard responses. GitHub Actions runs the suite on Windows and Linux with
Python 3.10, 3.11, and 3.12. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the development
verification record and limitations.

## Troubleshooting

- **Coinbase timeout / 403 / 429:** check outbound HTTPS access to
  `api.exchange.coinbase.com`. The client retries transient failures with backoff.
  Keep your cache and restart later. No API key is needed, and a key won't fix a
  blocked network. The offline demo can verify your installation separately.
- **Missing candles:** the provider can omit intervals with no trades. Retry;
  if the gap persists, use a shorter history or a more liquid USD pair. Fly Trade
  intentionally will not train across an invented, forward-filled interval.
- **No trades:** check the selected threshold, last update time, and risk-halt
  status. A quiet wallet can be correct, especially with costly intraday turnover.
- **Settings changed:** select a new data directory to begin an independent run.
- **Dashboard port busy:** change `app.dashboard_port`, or use `run --no-dashboard`.
- **`venv` / Python error:** install venv support for the Python 3.10+ interpreter you use.
  You can create `.venv` manually as shown above.

## Limits of the simulator

Historical candles don't contain queue position, market depth, within-candle event
order, or the fill you would actually receive. Fixed spread/slippage and full
virtual fills approximate those effects. Forward paper orders use observed prices,
but don't consume liquidity or simulate partial fills, exact exchange increments,
latency, or exchange-side stop orders. This is intraday spot research, not a
high-frequency execution system. Polling stops can miss price excursions between
observations; disconnects stop all execution until usable data returns.

The risk/execution boundary is isolated in `engine.py` and `paper.py`. A future live
adapter would need exchange-specific reconciliation, precision/minimum handling,
idempotent orders, partial fills, durable order tracking, and tested failure
recovery. Changing a config flag cannot turn this version into a real-money bot.
