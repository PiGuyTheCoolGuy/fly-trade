# Sources and model provenance

Market interface documentation inspected September 13, 2026:

- [Coinbase Exchange candle endpoint](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles): time buckets, supported granularities, 300-candle maximum, and incomplete-history behavior.
- [Coinbase Exchange product book](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book): level-1 bid/ask, quote timestamps, and auction-mode semantics.
- [Coinbase Exchange rate limits](https://docs.cdp.coinbase.com/exchange/rest-api/rate-limits): public endpoint throttling and HTTP 429.
- [Coinbase Exchange introduction](https://docs.cdp.coinbase.com/exchange/introduction/welcome): separation of public market data and authenticated trading APIs.

The downloader uses 299 buckets per page to leave room for inclusive endpoint
behavior, then filters to `[start, end)`. Data is fetched directly when you run the
program. Access/availability is determined by Coinbase and your network.

Optional biological connectivity:

- [Philip Shiu's Drosophila brain model repository](https://github.com/philshiu/Drosophila_brain_model).
- Pinned revision: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.
- Files: `Completeness_783.csv` (3,327,347 bytes) and `Connectivity_783.parquet`
  (100,804,642 bytes). Exact Git blob checksums are embedded in `connectome.py`.
- [Shiu et al. (2024), A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9).
- [FlyWire annotations and dataset context](https://github.com/flyconnectome/flywire_annotations).

Credit for the biological data belongs to the FlyWire community and the upstream
researchers. See the upstream repository, papers, and data terms for reuse and
citation requirements. Fly Trade's license covers its original application code;
it does not relicense third-party datasets. Downloaded data is not committed here.

The FlyWire adapter preserves the direction and sign of retained connections,
sums duplicates, normalizes each row's absolute weight to at most 0.8, injects
artificial market inputs, and pools graph activations. The market mapping,
normalization, neuron subset selection, simplified dynamics, and learned trading
readout are original experimental choices. They are not findings from the Shiu
paper and do not reproduce its simulations.

The default mushroom model uses synthetic wiring. No measured fly connectome is
downloaded or used in that mode. Neither cited neuroscience work nor this software
establishes a trading advantage from biological connectivity.
