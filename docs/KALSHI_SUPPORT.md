# Kalshi Support

This branch adds read-only Kalshi market-data support for paper/audit scans.

## What Works

- `poly-arb paper-arb-sim --exchange kalshi`
- Kalshi REST orderbook polling
- Paper-mode pair and N-leg scanning
- Audit mode using the existing forensic audit tables
- Kalshi YES/NO bid books normalized into the existing internal `OrderBook` model

Kalshi orderbooks expose YES and NO bids. The simulator derives asks from the opposite side:

- YES ask = `1 - best NO bid`
- NO ask = `1 - best YES bid`

## What Is Intentionally Not Enabled Yet

- Kalshi live order placement
- Kalshi WebSocket orderbook streaming
- Kalshi market discovery equivalent to Polymarket Gamma crypto discovery
- Dynamic watchlist building for Kalshi

Live mode remains blocked for Kalshi until there is a separate authenticated Kalshi order executor with tests.

## Pair YAML Format

Use `parent_market_ticker` and `child_market_ticker` for Kalshi markets:

```yaml
pairs:
  - name: KALSHI_EXAMPLE_PAIR
    enabled: true
    parent_market_ticker: KXEXAMPLE-26JAN01-T50
    child_market_ticker: KXEXAMPLE-26JAN01-B50
    parent_outcome_label: "Above 50"
    child_outcome_label: "Below 50"
    relation: child_implies_parent
    relation_subtype: kalshi_manual
    relation_safety: clean
    trade_template:
      leg_1:
        market: parent
        outcome: YES
        side: BUY
      leg_2:
        market: child
        outcome: NO
        side: BUY
    overrides:
      estimated_fee_rate: 0.0
      slippage_buffer: 0.0
```

The loader converts those tickers into internal synthetic token IDs:

```text
kalshi:<ticker>:yes
kalshi:<ticker>:no
```

You can also provide those synthetic IDs directly in the existing token fields.

## Example Command

```bash
poly-arb paper-arb-sim \
  --exchange kalshi \
  --book-source polling \
  --scan-all-candidates \
  --candidate-pairs config/kalshi_pairs.yaml \
  --entry-rest-recheck true \
  --budget 10000 \
  --max-trade-size 1000 \
  --max-total-locked-capital 10000 \
  --audit-mode forensic \
  --audit-dir reports/audit
```

If `--book-source websocket` is left as the default, the CLI currently switches Kalshi to REST polling and prints a note.

## Discover Kalshi Candidate YAML

```bash
poly-arb discover-kalshi-pairs \
  --status open \
  --limit 1000 \
  --max-pages 5 \
  --max-markets 100 \
  --min-volume 100 \
  --enabled \
  --out config/generated_kalshi_pairs.yaml
```

By default, discovery keeps only markets with both YES and NO quoted asks in the Kalshi market list. Use `--allow-unquoted` if you want a broad research universe that may mostly reject as missing ask/depth during scanning.

Discovery also validates the detailed orderbook by default and keeps only markets where the same-market YES+NO complement has executable implied ask depth on both legs. Use `--skip-orderbook-depth-check` only for broad offline research exports; those candidates may be rejected during scanning as `missing_ask` or `no_ask_depth`.
