# Architecture

Polybitrage is built around a single-threaded asyncio scan loop fed by a
multi-threaded Rust market-data client. The design goal is to keep the Python
event loop free of blocking work so the tick-to-trade path stays short and
predictable.

## Data flow, top to bottom

1. **Market data (Rust).** A dedicated Rust extension (`polymarket_rs`) owns a
   multi-threaded Tokio runtime with two redundant WebSocket connections to the
   Polymarket CLOB. Frames are parsed with zero-copy `serde` and applied to a
   lock-free `DashMap<token_id, BookEntry>`. Updated tokens are pushed onto a
   condvar-backed queue.
2. **Hand-off (GIL-released).** Python calls `poll_updates(timeout_ms)` on an
   asyncio executor thread. The call blocks on the Rust condvar with the GIL
   released, so the event loop is never stalled while waiting for market data.
3. **Scan loop (Python + native kernel).** Each iteration fetches only the books
   that changed, fills numpy price arrays for the dirty set, and runs the
   compiled C/Cython classifier over the full array. Near-arb and open-position
   pairs get a closer look; the depth-aware optimizer sizes entries.
4. **Execution.** Candidates are re-checked against REST before a paper or
   guarded live entry is recorded.
5. **Telemetry.** A health/metrics server exposes Prometheus metrics; snapshots
   and reports are written on background threads so disk I/O never touches the
   hot path.

## Runtime diagram

```text
──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    POLYMARKET CLOB                                          │
│     wss://ws-subscriptions-clob.polymarket.com/ws/market      https://clob.polymarket.com   │
└──────────────────────┬──────────────────────┬──────────────────────────────┬────────────────┘
           TLS WS #1   │        TLS WS #2     │                   HTTPS REST │
                       │                      │                              │
╔══════════════════════╪══════════════════════╪══════════════════════════════╪═════════════════╗
║          RUST EXTENSION — polymarket_rs  (tokio multi-threaded runtime)                      ║
║                      │                      │                                                ║
║   ┌───────────────────▼──────────┐  ┌────────▼──────────────────────┐                        ║
║   │  WS Slot 0  (tokio::spawn)   │  │  WS Slot 1  (tokio::spawn)    │  both subscribe to     ║
║   │  • connect / auto-reconnect  │  │  • connect / auto-reconnect   │  all tokens (HA        ║
║   │  • tokio-tungstenite frames  │  │  • tokio-tungstenite frames   │  redundancy; stagger   ║
║   │  • serde zero-copy parse     │  │  • serde zero-copy parse      │  slot*1000ms delay)    ║
║   │  • apply_snapshot/delta/bba  │  │  • apply_snapshot/delta/bba   │                        ║
║   └──────────────┬───────────────┘  └────────────────┬──────────────┘                        ║
║                  │   write via parking_lot::RwLock   │                                       ║
║                  └────────────────┬──────────────────┘                                       ║
║                                   ▼                                                          ║
║   ┌───────────────────────────────────────────────────────────────────────────────────────┐  ║
║   │  DashMap<token_id → BookEntry>                        (lock-free concurrent HashMap)  │  ║
║   │                                                                                       │  ║
║   │   BookEntry {                                                                         │  ║
║   │     bids : RwLock<BTreeMap<OrderedFloat(−price), size>>   ← highest bid = first key   │  ║
║   │     asks : RwLock<BTreeMap<OrderedFloat(+price), size>>   ← lowest ask  = first key   │  ║
║   │     last_updated : Mutex<Instant>                                                     │  ║
║   │   }                                                                                   │  ║
║   └───────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                   │ push (Mutex<Vec>)                                        ║
║                                   ▼                                                          ║
║   ┌───────────────────────────────────────────────────────────────────────────────────────┐  ║
║   │  UpdateQueue { pending: Mutex<Vec<token_id>>,  notify: Condvar }                      │  ║
║   └───────────────────────────────────────────────────────────────────────────────────────┘  ║
║                  ╎  condvar.wait_for(timeout_ms)   — GIL released while waiting              ║
║                  ╎  condvar.notify_one()            — called from WS slot tasks above        ║
╚══════════════════╪═══════════════════════════════════════════════════════════════════════════╝
                   ╎  (OS thread blocked here)
╔══════════════════╪═══════════════════════════════════════════════════════════════════════════╗
║                  ╎          OS THREAD POOL  (asyncio default executor)                       ║
║                  ╎                                                                           ║
║   Thread A:  poll_updates(timeout_ms) ───╎──────────────► blocks on condvar above            ║
║              ← returns Vec<token_id>  ◄──╯   (wakes on WS event OR timeout)                  ║
║                                                                                              ║
║   Thread B:  _write_markdown()   [asyncio.to_thread, every 5 s]   → .md file on disk         ║
║   Thread C:  save_snapshot()     [asyncio.to_thread, every 60 s]  → .json snapshot on disk   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
                   │  Future resolves with dirty-token list
                   ▼
╔═════════════════════════════════════════════════════════════════════════════════════════════╗
║           ASYNCIO EVENT LOOP  (main OS thread, single-threaded — no locks needed)           ║
║                                                                                             ║
║   ┌─── Main scan loop ─────────────────────────────────────────────────────────────────┐    ║
║   │                                                                                    │    ║
║   │  await provider.wait_for_updates(timeout=0.05s)                                    │    ║
║   │     └─ run_in_executor(None, rust.poll_updates, 50)  → dispatches to Thread A      │    ║
║   │        ← returns updated_tokens: set[str]            ← Future.result()             │    ║
║   │                                                                                    │    ║
║   │  _scan_once(pairs, provider, updated_tokens)                                       │    ║
║   │   ├─ compute dirty_idxs + open_idxs + needed_token_ids                             │    ║
║   │   │                                                                                │    ║
║   │   ├─ await provider.get_books(needed_token_ids)                                    │    ║
║   │   │    WebSocketOrderBookProvider                                                  │    ║
║   │   │     ├─ RustClobWsClient.get_book(token_id)  → DashMap read (each token)        │    ║
║   │   │     │    └─ get_book_snapshot() → from_float_data() → Python OrderBook         │    ║
║   │   │     └─ stale / missing → PollingOrderBookProvider fallback                     │    ║
║   │   │          └─ await ClobClient.get_order_books([...])  → HTTPS REST ──────────────►   ║
║   │   │               └─ ws_client.seed_books(rest_books)  → DashMap write             │    ║
║   │   │                                                                                │    ║
║   │   ├─ _fast_observe_all(books, dirty_idxs)         (pure Python + numpy)            │    ║
║   │   │    ├─ fill price arrays for dirty_idxs only  O(|dirty|) dict lookups           │    ║
║   │   │    ├─ _scan_classify_batch(...)               C/numba kernel (full array)      │    ║
║   │   │    │    └─ skipped entirely when dirty_idxs == ∅ (WS timeout, no open pos.)    │    ║
║   │   │    └─ _observe_pair()  only for near-arb / open positions                      │    ║
║   │   │                                                                                │    ║
║   │   ├─ _best_n_leg_opportunity()   numpy vectorised (skipped if n_leg_dirty=False)   │    ║
║   │   │                                                                                │    ║
║   │   └─ _try_enter / _try_enter_n_leg                                                 │    ║
║   │         └─ await ClobClient.get_order_books([candidate tokens])  → REST recheck ── ╫──► ║
║   │                                                                                    │    ║
║   └────────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                             ║
║   ┌─── Fire-and-forget asyncio Tasks  (cooperative; scheduled from scan loop) ───────────┐  ║
║   │                                                                                      │  ║
║   │  _bg_save_book_snapshot  (every 60 s)                                                │  ║
║   │    └─ asyncio.to_thread(ws_client.save_snapshot) ──────────────────────► Thread C    │  ║
║   │                                                                                      │  ║
║   │  _bg_enter_pair / _bg_enter_n_leg  (spawned when background_entry=True)              │  ║
║   │    ├─ reserves capital in _inflight_entries    ← shared dict, no lock (same thread)  │  ║
║   │    ├─ await REST recheck                      → HTTPS ────────────────────────────►  │  ║
║   │    └─ writes result into _pending_bg_actions  ← drained at next scan iteration       │  ║
║   │                                                                                      │  ║
║   │  priority_seed  (triggered when near-arb detected, throttled ≥ 2 s)                  │  ║
║   │    └─ await PollingProvider.get_books(near_arb_tokens)  → REST ──────────────────►   │  ║
║   │         └─ ws_client.seed_books(...)          → DashMap write (refresh stale book)   │  ║
║   │                                                                                      │  ║
║   │  asyncio.to_thread(_write_markdown)  (every 5 s) ─────────────────────► Thread B     │  ║
║   └──────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                             ║
║   ┌─── HealthServer  (asyncio HTTP, GET /health → port 8765) ─────────────────────────┐     ║
║   └───────────────────────────────────────────────────────────────────────────────────┘     ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝```

## Why this layout is fast

- **No blocking on the event loop.** Waiting for market data happens on an
  executor thread with the GIL released; report writing and snapshotting are
  offloaded with `asyncio.to_thread`.
- **Work proportional to change.** Only tokens whose books moved are re-priced;
  when nothing changes, the classifier is skipped entirely.
- **Native hot path.** The per-scan classifier is compiled C (via Cython) with
  `-O3 -march=native`; it falls back to a pure-numpy implementation if the
  extension is not built.
- **Redundant feeds.** Two staggered WebSocket connections provide failover; a
  REST polling provider seeds and backstops the cache when the stream is stale.
