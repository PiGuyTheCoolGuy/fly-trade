# Development verification

The application was exercised locally with Python 3.10 and 3.12 on Linux. The automated
suite uses explicit fake market responses and generated price histories; these are
software tests, not financial performance evidence.

- 38 automated tests passed on both Python versions across accounting, historical execution, leakage
  prevention, market parsing, persistence, restart/outage handling, and the full
  train/test/report/dashboard workflow.
- Reproduced Coinbase's nanosecond timestamp parsing failure on Python 3.10 before
  the fix. Regression tests now accept all eight reported timestamps, variable
  fractional precision, and explicit timezone offsets. Malformed, timezone-less,
  stale, and future quotes still fail validation.
- The 1,600-bar, three-market synthetic command-line demo completed, producing
  separate evaluation/deployment models, an HTML/JSON report, data snapshots, fills,
  and equity exports.
- Frozen-test reproduction matched its saved result using persisted candles and
  evaluation weights.
- The pinned FlyWire v783 files downloaded successfully and passed checksum
  verification. The 138,639-neuron source produced a 2,048-neuron subgraph with
  159,635 nonzero directed connections. This is a simplified graph adapter,
  not a reproduction of the upstream biological simulation.
  A complete 1,200-bar synthetic training/evaluation run also completed using
  this downloaded graph.
- Coinbase's documented endpoint formats were checked. Direct requests to the
  live Coinbase endpoint timed out in the development workspace, so successful
  live Coinbase downloads and a sustained live forward-paper session were **not
  verified here**. The program's live adapter was tested using controlled responses.
- Windows launch scripts and Docker/systemd examples are provided; Python 3.10,
  3.11, and 3.12 on Windows and Linux are configured in GitHub Actions. A locally passing suite is not a
  claim that every deployment environment was exercised.

No real-money trading was performed. No historical-market profitability claim is
made. The synthetic demo is separately labeled and cannot initialize a normal
forward paper wallet.
