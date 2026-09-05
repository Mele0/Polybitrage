//! polymarket_rs — zero-copy Rust WebSocket client for Polymarket CLOB data.
//!
//! Architecture
//! ============
//! * Two tokio tasks run in a dedicated multi-threaded runtime (owned by the
//!   Python object), each maintaining one WebSocket connection.
//! * JSON is parsed with serde's **borrowed** deserializer: string fields like
//!   token IDs and prices are `&'a str` slices into the original frame buffer,
//!   avoiding heap allocation in the hot path.
//! * The order book is maintained in a `DashMap<token_id, BookEntry>`.
//!   Each `BookEntry` has a `parking_lot::RwLock<BTreeMap<...>>` for bids and
//!   asks.  The `BTreeMap` key for bids is `OrderedFloat(-price)` so that
//!   natural ascending iteration gives descending price order.
//! * Updated token IDs are pushed into an `UpdateQueue` (Mutex + Condvar).
//!   Python calls `poll_updates(timeout_ms)` which blocks the OS thread until
//!   data arrives or the timeout expires.  This is called via
//!   `asyncio.get_event_loop().run_in_executor()` so the main loop is never
//!   blocked.
//!
//! Wire protocol (Polymarket CLOB WebSocket)
//! ==========================================
//! Subscription:
//!   `{"type":"market","assets_ids":[...],"custom_feature_enabled":true}`
//!
//! Incoming message types:
//!   "book"          — full snapshot: `{event_type, asset_id, bids:[{price,size}], asks:[...]}`
//!   "price_change"  — delta:        `{event_type, price_changes:[{asset_id, side, price, size}]}`
//!   "best_bid_ask"  — top update:   `{event_type, asset_id, best_bid, best_ask}`

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use ordered_float::OrderedFloat;
use parking_lot::{Condvar, Mutex, RwLock};
use pyo3::prelude::*;
use serde::Deserialize;
use tokio::runtime::Runtime;

// ── Order-book entry ──────────────────────────────────────────────────────────

/// Per-token order book.
///
/// Bids use `OrderedFloat(-price)` as key so BTreeMap natural order ≡
/// descending price.  Asks use `OrderedFloat(price)`, ascending.
struct BookEntry {
    bids: RwLock<BTreeMap<OrderedFloat<f64>, f64>>,
    asks: RwLock<BTreeMap<OrderedFloat<f64>, f64>>,
    last_updated: Mutex<Instant>,
    /// Exchange-event → local-receipt latency (ms) of the most recent message
    /// that touched this token.  `-1.0` means "unknown" (no parseable
    /// `timestamp` field seen yet).  This is the wire-latency signal the Python
    /// client exposes via `last_latency_ms`, now available on the Rust path too.
    last_latency_ms: Mutex<f64>,
}

impl BookEntry {
    fn new() -> Self {
        Self {
            bids: RwLock::new(BTreeMap::new()),
            asks: RwLock::new(BTreeMap::new()),
            last_updated: Mutex::new(Instant::now()),
            last_latency_ms: Mutex::new(-1.0),
        }
    }

    fn age_ms(&self) -> f64 {
        self.last_updated.lock().elapsed().as_secs_f64() * 1000.0
    }

    fn touch(&self) {
        *self.last_updated.lock() = Instant::now();
    }

    /// Record the wire latency of the message that just updated this book.
    fn set_latency(&self, lat: Option<f64>) {
        if let Some(l) = lat {
            *self.last_latency_ms.lock() = l;
        }
    }
}

// ── Update queue (Condvar-based; Python polls this) ───────────────────────────

struct UpdateQueue {
    /// Pending token IDs plus the `Instant` the *oldest* still-undrained item
    /// was enqueued.  Bundling the timestamp into the same mutex as the vector
    /// keeps the hot push/drain to a single lock and makes the queue-wait
    /// measurement exact (no separate lock to race).
    pending: Mutex<(Vec<String>, Option<Instant>)>,
    notify: Condvar,
}

impl UpdateQueue {
    fn new() -> Self {
        Self {
            pending: Mutex::new((Vec::new(), None)),
            notify: Condvar::new(),
        }
    }

    /// Push a batch of updated token IDs and wake one waiter.
    ///
    /// The counter increment lives here (not in parse_and_update) so that
    /// messages processed by both WS slots are not double-counted: the queue
    /// is the single merge point for both slots, so incrementing here gives a
    /// count that matches the Python dirty-set cardinality.
    fn push(&self, tokens: Vec<String>, count: &AtomicU64) {
        if tokens.is_empty() {
            return;
        }
        let n = tokens.len() as u64;
        let mut g = self.pending.lock();
        // Stamp the enqueue time on the empty→non-empty transition; while items
        // remain undrained the oldest stamp is preserved, so drain() can report
        // how long the *oldest* pending update waited for the main loop.
        if g.0.is_empty() {
            g.1 = Some(Instant::now());
        }
        g.0.extend(tokens);
        self.notify.notify_one();
        count.fetch_add(n, Ordering::Relaxed);
    }

    /// Block up to `timeout_ms`, then drain and deduplicate all pending IDs.
    /// Returns `(unique_ids, oldest_wait_ms)`; `oldest_wait_ms` is `-1.0` on a
    /// timeout with no data (the "queue was empty" sentinel).
    fn drain(&self, timeout_ms: u64) -> (Vec<String>, f64) {
        let mut g = self.pending.lock();
        if g.0.is_empty() {
            self.notify
                .wait_for(&mut g, Duration::from_millis(timeout_ms));
        }
        if g.0.is_empty() {
            return (Vec::new(), -1.0);
        }
        let wait_ms = g
            .1
            .take()
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(-1.0);
        // Drain, dedup without sorting (preserve first-seen order).
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(g.0.len());
        for tok in g.0.drain(..) {
            if seen.insert(tok.clone()) {
                out.push(tok);
            }
        }
        (out, wait_ms)
    }
}

// ── Shared WS state (lives for the lifetime of the tokio tasks) ───────────────

struct WsState {
    books: DashMap<String, BookEntry>,
    queue: UpdateQueue,
    tokens: RwLock<Vec<String>>,
    stop: AtomicBool,
    /// Per-slot resub flag; set by Python, cleared by the WS loop.
    resub: [AtomicBool; 2],
    connected_slots: AtomicI32,
    reconnect_count: AtomicU64,
    token_update_count: AtomicU64,
    // ── Deep WS telemetry (cheap relaxed atomics) ──
    /// Per-event-type message counts.
    book_msgs: AtomicU64,
    price_change_msgs: AtomicU64,
    bba_msgs: AtomicU64,
    /// Total decoded WS frames and their byte volume (feed throughput).
    frames_recv: AtomicU64,
    bytes_recv: AtomicU64,
    /// Frames that failed to parse (data-quality signal).
    parse_errors: AtomicU64,
    /// Monotonic instant of the last frame on any slot — for feed-staleness.
    last_msg_at: Mutex<Instant>,
}

impl WsState {
    fn new() -> Self {
        Self {
            books: DashMap::new(),
            queue: UpdateQueue::new(),
            tokens: RwLock::new(Vec::new()),
            stop: AtomicBool::new(false),
            resub: [AtomicBool::new(false), AtomicBool::new(false)],
            connected_slots: AtomicI32::new(0),
            reconnect_count: AtomicU64::new(0),
            token_update_count: AtomicU64::new(0),
            book_msgs: AtomicU64::new(0),
            price_change_msgs: AtomicU64::new(0),
            bba_msgs: AtomicU64::new(0),
            frames_recv: AtomicU64::new(0),
            bytes_recv: AtomicU64::new(0),
            parse_errors: AtomicU64::new(0),
            last_msg_at: Mutex::new(Instant::now()),
        }
    }
}

// ── serde types — borrow from the raw frame (zero-copy) ───────────────────────

/// A single WebSocket envelope.  All string fields are borrowed `&'a str`
/// slices into the original UTF-8 frame, so no heap allocation for these.
#[derive(Deserialize, Default)]
struct Envelope<'a> {
    #[serde(rename = "event_type", default)]
    event_type: Option<&'a str>,
    #[serde(rename = "type", default)]
    msg_type: Option<&'a str>,
    #[serde(rename = "asset_id", default)]
    asset_id: Option<&'a str>,
    #[serde(rename = "assetId", default)]
    asset_id2: Option<&'a str>,
    #[serde(default)]
    bids: Vec<Level<'a>>,
    #[serde(default)]
    asks: Vec<Level<'a>>,
    #[serde(default)]
    price_changes: Vec<PriceChange<'a>>,
    #[serde(rename = "best_bid", default)]
    best_bid: Option<&'a str>,
    #[serde(rename = "bestBid", default)]
    best_bid2: Option<&'a str>,
    #[serde(rename = "best_ask", default)]
    best_ask: Option<&'a str>,
    #[serde(rename = "bestAsk", default)]
    best_ask2: Option<&'a str>,
    /// Exchange-side event time (ms or s since epoch, as a string).  Used to
    /// compute wire latency.  Borrowed; no allocation.
    #[serde(rename = "timestamp", default)]
    timestamp: Option<&'a str>,
}

impl<'a> Envelope<'a> {
    fn event(&self) -> &str {
        self.event_type.or(self.msg_type).unwrap_or("")
    }
    fn token_id(&self) -> Option<&str> {
        self.asset_id.or(self.asset_id2)
    }
    fn best_bid_str(&self) -> Option<&str> {
        self.best_bid.or(self.best_bid2)
    }
    fn best_ask_str(&self) -> Option<&str> {
        self.best_ask.or(self.best_ask2)
    }
}

#[derive(Deserialize)]
struct Level<'a> {
    price: &'a str,
    size: &'a str,
}

#[derive(Deserialize, Default)]
struct PriceChange<'a> {
    #[serde(rename = "asset_id", default)]
    asset_id: Option<&'a str>,
    #[serde(rename = "assetId", default)]
    asset_id2: Option<&'a str>,
    #[serde(default)]
    side: Option<&'a str>,
    #[serde(default)]
    price: Option<&'a str>,
    #[serde(default)]
    size: Option<&'a str>,
}

impl<'a> PriceChange<'a> {
    fn token_id(&self) -> Option<&str> {
        self.asset_id.or(self.asset_id2)
    }
}

// ── Book mutation helpers ─────────────────────────────────────────────────────

fn parse_f64(s: Option<&str>) -> Option<f64> {
    s?.parse().ok()
}

/// Current wall-clock time in milliseconds since the Unix epoch.
fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Wire latency (ms) = local receipt time − exchange event `timestamp`.
///
/// Mirrors the Python `_message_latency_ms` guard exactly: the field may be in
/// milliseconds (when > 1e10) or seconds; out-of-range values (clock skew, bad
/// data) are discarded; negative is clamped to 0.  Returns `None` when there is
/// no usable timestamp so callers leave the stored latency unchanged.
fn wire_latency_ms(ts: Option<&str>) -> Option<f64> {
    let raw: i64 = ts?.trim().parse().ok()?;
    let ts_ms = if raw > 10_000_000_000 { raw } else { raw * 1000 };
    let lat = (now_unix_ms() - ts_ms) as f64;
    if !(-1000.0..=60_000.0).contains(&lat) {
        return None;
    }
    Some(lat.max(0.0))
}

fn apply_snapshot(tid: &str, bids: &[Level<'_>], asks: &[Level<'_>], lat: Option<f64>, books: &DashMap<String, BookEntry>) {
    let entry = books.entry(tid.to_owned()).or_insert_with(BookEntry::new);
    {
        let mut bm = entry.bids.write();
        bm.clear();
        for lvl in bids {
            if let (Ok(p), Ok(s)) = (lvl.price.parse::<f64>(), lvl.size.parse::<f64>()) {
                if s > 0.0 {
                    bm.insert(OrderedFloat(-p), s);
                }
            }
        }
    }
    {
        let mut am = entry.asks.write();
        am.clear();
        for lvl in asks {
            if let (Ok(p), Ok(s)) = (lvl.price.parse::<f64>(), lvl.size.parse::<f64>()) {
                if s > 0.0 {
                    am.insert(OrderedFloat(p), s);
                }
            }
        }
    }
    entry.touch();
    entry.set_latency(lat);
}

fn apply_delta(tid: &str, change: &PriceChange<'_>, lat: Option<f64>, books: &DashMap<String, BookEntry>) {
    let p = match parse_f64(change.price) { Some(v) => v, None => return };
    let s = match parse_f64(change.size)  { Some(v) => v, None => return };
    let is_buy = change.side.map(|x| x.eq_ignore_ascii_case("BUY")).unwrap_or(false);
    let entry = books.entry(tid.to_owned()).or_insert_with(BookEntry::new);
    // Only touch (refresh age) when the book actually changed.  A zero-size
    // delta for a price level that no longer exists is a no-op; touching in
    // that case would mask real staleness from max_book_age_ms tracking.
    let changed = if is_buy {
        let k = OrderedFloat(-p);
        let mut bm = entry.bids.write();
        if s > 0.0 { bm.insert(k, s); true } else { bm.remove(&k).is_some() }
    } else {
        let k = OrderedFloat(p);
        let mut am = entry.asks.write();
        if s > 0.0 { am.insert(k, s); true } else { am.remove(&k).is_some() }
    };
    if changed {
        entry.touch();
        entry.set_latency(lat);
    }
}

/// Update the best bid / ask from a best_bid_ask WS event.
///
/// Two bugs fixed vs. the old implementation:
///
/// 1. **Fake liquidity**: the old code carried the old top-of-book size to the
///    new price.  A best_bid_ask event carries no depth; that carried size is
///    invented.  We now use `entry().or_insert(0.0)` so an unknown size is
///    recorded as 0.0, which the Python scanner's depth filter correctly treats
///    as "no known depth".
///
/// 2. **O(n) retain**: `BTreeMap::retain` visits every entry.  We replace it
///    with an O(k) forward loop that pops from the cheap front of the map until
///    the stale prefix is gone (k ≈ 0–1 in practice).
fn apply_bba(tid: &str, env: &Envelope<'_>, lat: Option<f64>, books: &DashMap<String, BookEntry>) -> bool {
    let best_bid = parse_f64(env.best_bid_str()).filter(|&p| p > 0.0);
    let best_ask = parse_f64(env.best_ask_str()).filter(|&p| p > 0.0);
    if best_bid.is_none() && best_ask.is_none() {
        return false;
    }
    let entry = books.entry(tid.to_owned()).or_insert_with(BookEntry::new);

    if let Some(fp) = best_bid {
        // Bids stored as OrderedFloat(-price); ascending BTreeMap order ≡
        // descending price.  Stale bids have price > fp → key < OrderedFloat(-fp).
        let nk = OrderedFloat(-fp);
        let mut bm = entry.bids.write();
        // O(k) prune using pop_first() (stable since Rust 1.66): single
        // BTree traversal per removed entry vs. iter().next() + remove() = two.
        while bm.first_key_value().map_or(false, |(&k, _)| k < nk) {
            bm.pop_first();
        }
        // Insert at new price only if no real depth exists there yet.
        // or_insert keeps an existing size; 0.0 signals "unknown depth".
        bm.entry(nk).or_insert(0.0);
    }

    if let Some(fp) = best_ask {
        // Asks stored as OrderedFloat(price); ascending BTreeMap order ≡
        // ascending price.  Stale asks have price < fp → key < OrderedFloat(fp).
        let ak = OrderedFloat(fp);
        let mut am = entry.asks.write();
        // Same pop_first() pattern for the ask side.
        while am.first_key_value().map_or(false, |(&k, _)| k < ak) {
            am.pop_first();
        }
        am.entry(ak).or_insert(0.0);
    }

    entry.touch();
    entry.set_latency(lat);
    true
}

// ── Core message dispatcher ───────────────────────────────────────────────────

/// Parse one raw WS text frame and mutate `books` in-place.
/// Returns the set of updated token IDs.
fn parse_and_update(text: &str, state: &WsState) -> Vec<String> {
    // The server sends either a JSON array or a single object.
    // Use borrowed deserialization — string fields are `&str` into `text`.
    let envs: Vec<Envelope<'_>> = if text.trim_start().starts_with('[') {
        serde_json::from_str(text).unwrap_or_default()
    } else {
        match serde_json::from_str::<Envelope<'_>>(text) {
            Ok(e) => vec![e],
            Err(_) => {
                state.parse_errors.fetch_add(1, Ordering::Relaxed);
                return Vec::new();
            }
        }
    };

    let mut updated: Vec<String> = Vec::with_capacity(envs.len() * 2);

    for env in &envs {
        // One wall-clock read per envelope; the timestamp applies to every token
        // the envelope touches (price_change carries multiple tokens).
        let lat = wire_latency_ms(env.timestamp);
        match env.event() {
            "book" => {
                if let Some(tid) = env.token_id() {
                    apply_snapshot(tid, &env.bids, &env.asks, lat, &state.books);
                    updated.push(tid.to_owned());
                    state.book_msgs.fetch_add(1, Ordering::Relaxed);
                }
            }
            "price_change" => {
                for change in &env.price_changes {
                    if let Some(tid) = change.token_id() {
                        apply_delta(tid, change, lat, &state.books);
                        updated.push(tid.to_owned());
                    }
                }
                state.price_change_msgs.fetch_add(1, Ordering::Relaxed);
            }
            "best_bid_ask" => {
                if let Some(tid) = env.token_id() {
                    if apply_bba(tid, env, lat, &state.books) {
                        updated.push(tid.to_owned());
                    }
                }
                state.bba_msgs.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    if !updated.is_empty() {
        updated.sort_unstable();
        updated.dedup();
    }
    updated
}

// ── Subscription message builder ─────────────────────────────────────────────

fn make_sub_msg(tokens: &[String]) -> String {
    let mut sorted = tokens.to_vec();
    sorted.sort();
    serde_json::json!({
        "type": "market",
        "assets_ids": sorted,
        "custom_feature_enabled": true
    })
    .to_string()
}

// ── WebSocket connection loop (one tokio task per slot) ───────────────────────

async fn ws_slot_loop(slot: usize, ws_url: String, state: Arc<WsState>, delay_ms: u64) {
    use tokio_tungstenite::tungstenite::Message;

    // Stagger initial connect to avoid thundering herd
    if slot > 0 {
        tokio::time::sleep(Duration::from_millis(slot as u64 * 1000)).await;
    }

    'outer: loop {
        if state.stop.load(Ordering::Relaxed) {
            break;
        }

        let current_tokens: Vec<String> = state.tokens.read().clone();
        let connect_result = tokio_tungstenite::connect_async(ws_url.as_str()).await;

        match connect_result {
            Err(e) => {
                eprintln!("[polymarket_rs] WS[{slot}] connect error: {e}");
            }
            Ok((stream, _response)) => {
                let (mut write, mut read) = stream.split();

                state.connected_slots.fetch_add(1, Ordering::Relaxed);
                state.resub[slot].store(false, Ordering::Relaxed);

                // Initial subscription
                let sub = make_sub_msg(&current_tokens);
                if write.send(Message::Text(sub.into())).await.is_err() {
                    state.connected_slots.fetch_sub(1, Ordering::Relaxed);
                    break 'outer;
                }

                let mut check = tokio::time::interval(Duration::from_millis(500));
                check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                'recv: loop {
                    tokio::select! {
                        msg = read.next() => {
                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    state.frames_recv.fetch_add(1, Ordering::Relaxed);
                                    state.bytes_recv.fetch_add(text.len() as u64, Ordering::Relaxed);
                                    *state.last_msg_at.lock() = Instant::now();
                                    let updated = parse_and_update(&text, &state);
                                    state.queue.push(updated, &state.token_update_count);
                                }
                                Some(Ok(Message::Binary(data))) => {
                                    state.frames_recv.fetch_add(1, Ordering::Relaxed);
                                    state.bytes_recv.fetch_add(data.len() as u64, Ordering::Relaxed);
                                    *state.last_msg_at.lock() = Instant::now();
                                    if let Ok(s) = std::str::from_utf8(&data) {
                                        let updated = parse_and_update(s, &state);
                                        state.queue.push(updated, &state.token_update_count);
                                    }
                                }
                                Some(Ok(Message::Ping(d))) => {
                                    let _ = write.send(Message::Pong(d)).await;
                                }
                                Some(Ok(Message::Close(_))) | None => {
                                    break 'recv;
                                }
                                Some(Err(_)) => {
                                    break 'recv;
                                }
                                _ => {}
                            }
                        }
                        _ = check.tick() => {
                            if state.stop.load(Ordering::Relaxed) {
                                let _ = write.send(Message::Close(None)).await;
                                break 'outer;
                            }
                            if state.resub[slot].swap(false, Ordering::Relaxed) {
                                let new_toks: Vec<String> = state.tokens.read().clone();
                                let sub = make_sub_msg(&new_toks);
                                if write.send(Message::Text(sub.into())).await.is_err() {
                                    break 'recv;
                                }
                            }
                        }
                    }
                }

                state.connected_slots.fetch_sub(1, Ordering::Relaxed);
            }
        }

        if state.stop.load(Ordering::Relaxed) {
            break 'outer;
        }
        state.reconnect_count.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
}

// ── PyO3 bindings ─────────────────────────────────────────────────────────────

/// Zero-copy Rust WebSocket client for Polymarket CLOB market data.
///
/// Wraps two tokio async WS tasks running in a dedicated multi-threaded
/// runtime.  Use the Python wrapper ``RustClobWsClient`` (in
/// ``src/polymarket/rust_clob_ws_client.py``) for asyncio integration.
#[pyclass]
struct RustWsClient {
    state: Arc<WsState>,
    rt: Runtime,
    ws_url: String,
    reconnect_delay_ms: u64,
    started: AtomicBool,
    /// f64 bits of the most recent `poll_updates` queue-wait (ms); `-1.0` until
    /// the first non-empty drain.  Single-consumer, so a plain atomic suffices.
    last_queue_wait_ms: AtomicU64,
    /// Tokens returned by the most recent non-empty drain (backpressure depth).
    last_drain_size: AtomicU64,
}

impl Drop for RustWsClient {
    fn drop(&mut self) {
        // Signal tasks to stop so the runtime can shut down cleanly.
        self.state.stop.store(true, Ordering::SeqCst);
        self.state.queue.notify.notify_all();
    }
}

#[pymethods]
impl RustWsClient {
    /// Create a new client.  Does not connect yet; call ``start()`` first.
    #[new]
    #[pyo3(signature = (ws_url, reconnect_delay_s = 5.0))]
    fn new(ws_url: String, reconnect_delay_s: f64) -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("tokio runtime init failed: {e}"))
        })?;
        Ok(Self {
            state: Arc::new(WsState::new()),
            rt,
            ws_url,
            reconnect_delay_ms: (reconnect_delay_s * 1000.0).max(100.0) as u64,
            started: AtomicBool::new(false),
            last_queue_wait_ms: AtomicU64::new((-1.0_f64).to_bits()),
            last_drain_size: AtomicU64::new(0),
        })
    }

    /// Connect and subscribe to `token_ids`.
    /// Safe to call multiple times; after the first call it just updates
    /// the subscription.
    fn start(&self, token_ids: Vec<String>) {
        *self.state.tokens.write() = token_ids;
        self.state.stop.store(false, Ordering::SeqCst);

        if self
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // First start — spawn one task per connection slot.
            for slot in 0..2usize {
                let state = Arc::clone(&self.state);
                let url = self.ws_url.clone();
                let delay = self.reconnect_delay_ms;
                self.rt.spawn(async move {
                    ws_slot_loop(slot, url, state, delay).await;
                });
            }
        } else {
            // Already running — trigger re-subscription on all slots.
            self.state.resub[0].store(true, Ordering::Relaxed);
            self.state.resub[1].store(true, Ordering::Relaxed);
        }
    }

    /// Update the active subscriptions without reconnecting.
    fn update_subscriptions(&self, token_ids: Vec<String>) {
        *self.state.tokens.write() = token_ids;
        self.state.resub[0].store(true, Ordering::Relaxed);
        self.state.resub[1].store(true, Ordering::Relaxed);
    }

    /// Signal both WS tasks to disconnect and stop.
    fn stop(&self) {
        self.state.stop.store(true, Ordering::SeqCst);
        self.state.queue.notify.notify_all();
        self.started.store(false, Ordering::SeqCst);
    }

    /// Block the *calling OS thread* for up to `timeout_ms` waiting for book
    /// updates.  Returns the list of updated token IDs (may be empty on
    /// timeout).  Call this via `loop.run_in_executor()` so the asyncio main
    /// loop is not blocked.
    fn poll_updates(&self, py: Python<'_>, timeout_ms: u64) -> Vec<String> {
        // CRITICAL: release the GIL for the blocking condvar wait inside drain().
        // Without py.detach (PyO3's GIL-release), this pymethod holds the GIL for
        // the whole timeout — so even though it runs in a run_in_executor thread,
        // it freezes the asyncio event loop (and every other task) for up to
        // `timeout_ms`.  The latency profiler caught this as loop_lag ≈ poll
        // timeout.  drain() touches only Rust state, so releasing is safe; the
        // GIL is re-acquired before we return the Vec to Python.
        let state = Arc::clone(&self.state);
        let (ids, wait_ms) = py.detach(move || state.queue.drain(timeout_ms));
        // Only publish the wait on a real drain; a timeout (sentinel -1.0) leaves
        // the previous value untouched so the Python side can ignore empties.
        if wait_ms >= 0.0 {
            self.last_queue_wait_ms
                .store(wait_ms.to_bits(), Ordering::Relaxed);
            self.last_drain_size
                .store(ids.len() as u64, Ordering::Relaxed);
        }
        ids
    }

    /// Tokens returned by the most recent non-empty drain — how many updates had
    /// piled up by the time the loop got to them (tick backpressure depth).
    fn last_drain_size(&self) -> u64 {
        self.last_drain_size.load(Ordering::Relaxed)
    }

    // ── Deep WS telemetry getters (read on the /metrics scrape) ──
    fn book_msg_count(&self) -> u64 { self.state.book_msgs.load(Ordering::Relaxed) }
    fn price_change_msg_count(&self) -> u64 { self.state.price_change_msgs.load(Ordering::Relaxed) }
    fn bba_msg_count(&self) -> u64 { self.state.bba_msgs.load(Ordering::Relaxed) }
    fn frames_received(&self) -> u64 { self.state.frames_recv.load(Ordering::Relaxed) }
    fn bytes_received(&self) -> u64 { self.state.bytes_recv.load(Ordering::Relaxed) }
    fn parse_error_count(&self) -> u64 { self.state.parse_errors.load(Ordering::Relaxed) }
    fn connected_slot_count(&self) -> i32 { self.state.connected_slots.load(Ordering::Relaxed) }

    /// Milliseconds since the last frame arrived on any slot — feed staleness.
    /// A rising value while markets are open means the feed (not a token) is quiet.
    fn ms_since_last_message(&self) -> f64 {
        self.state.last_msg_at.lock().elapsed().as_secs_f64() * 1000.0
    }

    /// Queue-wait (ms) of the oldest update returned by the most recent
    /// non-empty `poll_updates` — i.e. how long that update sat in the Rust
    /// queue before the asyncio loop drained it.  Rises when the loop is
    /// CPU-bound and falling behind the WS firehose.  `None` until first drain.
    fn last_queue_wait_ms(&self) -> Option<f64> {
        let v = f64::from_bits(self.last_queue_wait_ms.load(Ordering::Relaxed));
        if v < 0.0 {
            None
        } else {
            Some(v)
        }
    }

    /// Wire latency (ms) of the last message that touched `token_id`:
    /// exchange-event `timestamp` → local receipt.  `None` if the token is
    /// unknown or no parseable timestamp has been seen.
    fn get_update_latency_ms(&self, token_id: &str) -> Option<f64> {
        let v = *self.state.books.get(token_id)?.last_latency_ms.lock();
        if v < 0.0 {
            None
        } else {
            Some(v)
        }
    }

    /// Return `(bids, asks, age_ms)` for `token_id`, or `None` if unknown.
    ///
    /// `bids` — `list[tuple[float, float]]` sorted by price **descending**.
    /// `asks` — `list[tuple[float, float]]` sorted by price **ascending**.
    /// `age_ms` — milliseconds since the last update.
    fn get_book_snapshot(
        &self,
        token_id: &str,
    ) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>, f64)> {
        let entry = self.state.books.get(token_id)?;
        let bids: Vec<(f64, f64)> = entry
            .bids
            .read()
            .iter()
            .map(|(&k, &v)| (-k.0, v)) // un-negate key to recover price
            .collect();
        let asks: Vec<(f64, f64)> = entry
            .asks
            .read()
            .iter()
            .map(|(&k, &v)| (k.0, v))
            .collect();
        let age = entry.age_ms();
        Some((bids, asks, age))
    }

    /// Milliseconds since last update for `token_id`, or `None` if unknown.
    fn get_book_age_ms(&self, token_id: &str) -> Option<f64> {
        self.state.books.get(token_id).map(|e| e.age_ms())
    }

    /// Pre-warm the book for `token_id` with REST-fetched data (seed path).
    fn seed_book(&self, token_id: String, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        let entry = self
            .state
            .books
            .entry(token_id)
            .or_insert_with(BookEntry::new);
        {
            let mut bm = entry.bids.write();
            bm.clear();
            for (p, s) in &bids {
                if *s > 0.0 {
                    bm.insert(OrderedFloat(-p), *s);
                }
            }
        }
        {
            let mut am = entry.asks.write();
            am.clear();
            for (p, s) in &asks {
                if *s > 0.0 {
                    am.insert(OrderedFloat(*p), *s);
                }
            }
        }
        entry.touch();
    }

    /// All token IDs currently tracked (for snapshot saving).
    fn all_token_ids(&self) -> Vec<String> {
        self.state.books.iter().map(|r| r.key().clone()).collect()
    }

    /// True if at least one WS slot is connected.
    fn is_connected(&self) -> bool {
        self.state.connected_slots.load(Ordering::Relaxed) > 0
    }

    fn reconnect_count(&self) -> u64 {
        self.state.reconnect_count.load(Ordering::Relaxed)
    }

    fn token_update_count(&self) -> u64 {
        self.state.token_update_count.load(Ordering::Relaxed)
    }
}

// ── Module registration ────────────────────────────────────────────────────────

#[pymodule]
fn polymarket_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustWsClient>()?;
    Ok(())
}
