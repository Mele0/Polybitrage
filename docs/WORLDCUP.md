# World Cup arbitrage subsystem

Adapts the existing Polymarket pair / N-leg arbitrage engine to automatically
discover and trade **FIFA World Cup** markets — match props (moneyline, totals,
BTTS, spreads, exact score, corners, cards, player props) and tournament futures
ladders (winner, player goals, missed penalties).

It does **not** rewrite the bot. The core insight is that the existing engine is
relation-agnostic: every deterministic arb reduces to one of two primitives it
already prices, so the World Cup layer only adds *discovery* and *relation
inference* that emit those primitives, plus a *state validator* and *exposure
limiter* on top.

## How it maps onto the existing engine

| World Cup relation | Primitive | Existing machinery reused |
|---|---|---|
| Moneyline 3-way, O/U cross-market complement, quantity-vs-threshold | N-leg exhaustive partition, payout = 1 | depth-aware sizing math (`OrderBook.avg_ask_price_for_size`), worst-case = payout − cost |
| Totals / team-totals / corners / cards / shots / spread / N+ / exact-score ladders | 2-leg subset (`parent YES + child NO`) | `PairConfig` schema → consumable by `poly-arb paper-arb-sim` |

Order books, the CLOB client, the public WebSocket cache, and the guarded live
`LiveTrader` (FOK bundle posting) are all reused as-is.

## Modules (`src/worldcup/`)

- `schema.py` — normalized `WorldCupMarket` + market-type classifier (`sportsMarketType` + `line` + title regex).
- `discovery.py` — Gamma event enumeration (tag 102232 / series 11433), gameId grouping, JSON cache.
- `relations.py` — relation inference engine → `Relation` graph; emits engine-compatible `PairConfig` dicts.
- `state_validator.py` — enumerates terminal states; confirms worst-case payout > cost (the hard guard).
- `executor.py` — `ExposureLimiter` (kill switch + per-event/team/relation/daily caps) + paper/dry-run/live executor.
- `scanner.py` — orchestrates discover → infer → fetch books → size → validate → rank → execute → log.
- `cli.py` — `discover-worldcup` and `worldcup-scan` command runners.

## Commands

Discover events + relations (paper-only; writes a disabled subset-arb watchlist YAML):

```bash
poly-arb discover-worldcup --out-dir reports/worldcup
```

Scan once, evaluate only (never enters), print the table + per-relation logs:

```bash
poly-arb worldcup-scan --once --no-execute --show-top 15
```

Paper trading (default) — enters paper positions when a state-validated edge clears `min_edge_bps`:

```bash
poly-arb worldcup-scan --config config/worldcup.yaml --poll-seconds 3
```

Dry-run — prints exactly what *would* be bought, no positions:

```bash
poly-arb worldcup-scan --execution-mode dry_run --once
```

Live (guarded — reuses the same env vars / confirmation phrases as the rest of the bot):

```bash
export POLYMARKET_LIVE_TRADING_ENABLED=true
export POLYMARKET_PRIVATE_KEY=...   # + FUNDER/SIGNATURE_TYPE/API creds as in README
poly-arb worldcup-scan \
  --execution-mode live \
  --live-confirmation I_UNDERSTAND_THIS_USES_REAL_MONEY \
  --live-compliance-ack I_AM_ALLOWED_TO_TRADE_POLYMARKET \
  --live-max-session-spend 25 --max-trade-size 5
```

The subset-arb watchlist written by discovery can also be scanned by the original engine:

```bash
poly-arb paper-arb-sim --pairs reports/worldcup/generated_worldcup_pairs.yaml --include-disabled --once
```

## Config flags (`config/worldcup.yaml`, all overridable on the CLI)

`world_cup_mode`, `paper_trade`, `execution_mode`, `enable_deterministic_arbs`,
`enable_statistical_arbs`, `min_edge_bps` (MIN_EDGE_BPS), `max_leg_slippage`
(MAX_LEG_SLIPPAGE), `fee_rate`, `min_liquidity` (MIN_LIQUIDITY), `max_stale_ms`
(MAX_STALE_MS), `max_trade_size`, and exposure caps `max_event_exposure`,
`max_team_exposure`, `max_relation_exposure`, `max_total_exposure`,
`max_daily_loss`, plus `kill_switch_path` (create that file to halt all entries).

Manual relation overrides / market blacklist: copy `config/worldcup_overrides.example.yaml`
to `config/worldcup_overrides.yaml`.

## Outputs (`reports/worldcup/`)

`discovery_cache.json`, `events.json`, `relations.json`,
`generated_worldcup_pairs.yaml` (subset arbs for the original engine),
`opportunities.csv`, `skipped.csv` (with reason).

## Tests

```bash
pytest tests/test_worldcup_arbs.py -q
```

Covers complement-basket, over/under ladder, spread ladder, exact-score subset,
quantity-vs-threshold, the state-space validator, the no-false-arb guard, the
exposure limiter, and that an emitted `PairConfig` resolves through the original engine.

## Safety

Paper is the default. Deterministic arbs must pass the state-space validator
(worst-case payout strictly above executable cost after fees/slippage/buffer)
*and* every exposure cap before any entry. Live mode is FOK and is **not**
cross-leg atomic on Polymarket; on a partial bundle the executor surfaces the
error and does not commit exposure rather than chasing. Statistical edges are
kept separate from deterministic arbs and are off by default.
