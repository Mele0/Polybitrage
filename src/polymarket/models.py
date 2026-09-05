from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sortedcontainers import SortedDict

ZERO = Decimal("0")


def to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc


class OrderBookLevel:
    """Immutable price level. Behaves like a frozen dataclass for all existing code."""

    __slots__ = ("price", "size")

    def __init__(self, price: Decimal, size: Decimal) -> None:
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "size", size)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("OrderBookLevel is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderBookLevel):
            return NotImplemented
        return self.price == other.price and self.size == other.size

    def __hash__(self) -> int:
        return hash((self.price, self.size))

    def __repr__(self) -> str:
        return f"OrderBookLevel(price={self.price}, size={self.size})"

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "OrderBookLevel":
        return cls(price=to_decimal(payload.get("price")), size=to_decimal(payload.get("size")))


class OrderBook:
    """Order book with O(1) best-bid/ask and O(1) depth read.

    **Internal storage uses float keys and float values** for maximum arithmetic
    speed on the hot path (WS delta application, best-ask lookup, depth sum).
    Decimal is only used at the public API boundary (apply_*_delta signatures,
    OrderBookLevel list properties) so all existing callers remain compatible.

    bids: sorted descending by price (SortedDict with negated key).
    asks: sorted ascending by price (default SortedDict ordering).

    Four hot-path cached scalars are maintained incrementally so that reading
    them costs a single attribute lookup rather than a SortedDict traversal or
    a full O(n) sum:

        _best_bid   float | None  — top-of-book bid price
        _best_ask   float | None  — top-of-book ask price
        _bid_depth  float         — total (sum) bid size
        _ask_depth  float         — total (sum) ask size

    Correctness rules
    -----------------
    * ``apply_bid/ask_delta`` — incremental O(log n).  Updates all four
      scalars without a full recompute.
    * ``replace_best_bid/ask`` — called for WS best_bid_ask events.  Prunes
      stale levels in O(k) using ``peekitem``; does NOT carry the old top
      size to the new price (fake-liquidity fix).
    * ``from_float_data`` — zero-copy classmethod for the Rust WS hot path.
      Bypasses all Decimal conversion.  Accepts ``list[tuple[float, float]]``
      directly.
    """

    __slots__ = (
        "asset_id",
        "timestamp",
        "raw_json",
        "_bids_sd",
        "_asks_sd",
        "_bids_cache",
        "_asks_cache",
        "_bid_depth",   # float: incremental sum of bid sizes
        "_ask_depth",   # float: incremental sum of ask sizes
        "_best_bid",    # float | None: cached top bid price
        "_best_ask",    # float | None: cached top ask price
    )

    def __init__(
        self,
        asset_id: str = "",
        bids: list[OrderBookLevel] | None = None,
        asks: list[OrderBookLevel] | None = None,
        timestamp: datetime | None = None,
        raw_json: dict[str, Any] | None = None,
    ) -> None:
        self.asset_id = asset_id
        self.timestamp = timestamp
        self.raw_json: dict[str, Any] = raw_json if raw_json is not None else {}
        # Float keys for fast comparison; negated key keeps bids descending.
        self._bids_sd: SortedDict = SortedDict(lambda p: -p)
        self._asks_sd: SortedDict = SortedDict()
        self._bids_cache: list[OrderBookLevel] | None = None
        self._asks_cache: list[OrderBookLevel] | None = None
        # Cached scalars — maintained incrementally by all mutation paths.
        self._bid_depth: float = 0.0
        self._ask_depth: float = 0.0
        self._best_bid: float | None = None
        self._best_ask: float | None = None
        for level in (bids or []):
            p = float(level.price)
            s = float(level.size)
            self._bids_sd[p] = s
            self._bid_depth += s
            if self._best_bid is None or p > self._best_bid:
                self._best_bid = p
        for level in (asks or []):
            p = float(level.price)
            s = float(level.size)
            self._asks_sd[p] = s
            self._ask_depth += s
            if self._best_ask is None or p < self._best_ask:
                self._best_ask = p

    # ── Fast constructor for Rust WS path (no Decimal allocation) ────────────

    @classmethod
    def from_float_data(
        cls,
        asset_id: str,
        bids: list[tuple[float, float]],  # (price, size) sorted descending
        asks: list[tuple[float, float]],  # (price, size) sorted ascending
    ) -> "OrderBook":
        """Build an OrderBook directly from float tuples without any Decimal.

        Used by ``RustClobWsClient.get_book()`` to avoid the expensive
        Decimal(str(float)) round-trip on every WS-updated token.
        Levels with size == 0.0 are stored (preserving best-bid/ask price
        awareness from BBA events) but do not contribute to depth.
        """
        obj: "OrderBook" = cls.__new__(cls)
        obj.asset_id = asset_id
        obj.timestamp = None
        obj.raw_json = {}
        obj._bids_sd = SortedDict(lambda p: -p)
        obj._asks_sd = SortedDict()
        obj._bids_cache = None
        obj._asks_cache = None
        bd = 0.0
        best_b: float | None = None
        for p, s in bids:
            obj._bids_sd[p] = s
            if s > 0.0:
                bd += s
            # bids arrive sorted descending; first entry is best bid
            if best_b is None or p > best_b:
                best_b = p
        obj._bid_depth = bd
        obj._best_bid = best_b
        ad = 0.0
        best_a: float | None = None
        for p, s in asks:
            obj._asks_sd[p] = s
            if s > 0.0:
                ad += s
            # asks arrive sorted ascending; first entry is best ask
            if best_a is None or p < best_a:
                best_a = p
        obj._ask_depth = ad
        obj._best_ask = best_a
        return obj

    # ── bids property (descending) ────────────────────────────────────────────

    @property
    def bids(self) -> list[OrderBookLevel]:
        if self._bids_cache is None:
            self._bids_cache = [
                OrderBookLevel(price=Decimal(str(p)), size=Decimal(str(s)))
                for p, s in self._bids_sd.items()
            ]
        return self._bids_cache

    @bids.setter
    def bids(self, value: list[OrderBookLevel]) -> None:
        self._bids_sd.clear()
        bd = 0.0
        best_b: float | None = None
        for level in value:
            p = float(level.price)
            s = float(level.size)
            self._bids_sd[p] = s
            bd += s
            if best_b is None or p > best_b:
                best_b = p
        self._bid_depth = bd
        self._best_bid = best_b
        self._bids_cache = None

    # ── asks property (ascending) ─────────────────────────────────────────────

    @property
    def asks(self) -> list[OrderBookLevel]:
        if self._asks_cache is None:
            self._asks_cache = [
                OrderBookLevel(price=Decimal(str(p)), size=Decimal(str(s)))
                for p, s in self._asks_sd.items()
            ]
        return self._asks_cache

    @asks.setter
    def asks(self, value: list[OrderBookLevel]) -> None:
        self._asks_sd.clear()
        ad = 0.0
        best_a: float | None = None
        for level in value:
            p = float(level.price)
            s = float(level.size)
            self._asks_sd[p] = s
            ad += s
            if best_a is None or p < best_a:
                best_a = p
        self._ask_depth = ad
        self._best_ask = best_a
        self._asks_cache = None

    # ── O(log n) incremental delta application ────────────────────────────────

    def apply_bid_delta(self, price: Decimal, size: Decimal) -> None:
        """Apply one incremental bid update. O(log n). Updates depth and best_bid."""
        fp, fs = float(price), float(size)
        old = self._bids_sd.get(fp, 0.0)
        if fs > 0.0:
            if old == fs:
                return  # no change — avoid dirtying cache
            self._bids_sd[fp] = fs
            self._bid_depth += fs - old
            if self._best_bid is None or fp > self._best_bid:
                self._best_bid = fp
        else:
            removed = self._bids_sd.pop(fp, None)
            if removed is None:
                return
            self._bid_depth -= removed
            # Only recompute best_bid if we just removed it.
            if fp == self._best_bid:
                self._best_bid = self._bids_sd.peekitem(0)[0] if self._bids_sd else None
        self._bids_cache = None

    def apply_ask_delta(self, price: Decimal, size: Decimal) -> None:
        """Apply one incremental ask update. O(log n). Updates depth and best_ask."""
        fp, fs = float(price), float(size)
        old = self._asks_sd.get(fp, 0.0)
        if fs > 0.0:
            if old == fs:
                return
            self._asks_sd[fp] = fs
            self._ask_depth += fs - old
            if self._best_ask is None or fp < self._best_ask:
                self._best_ask = fp
        else:
            removed = self._asks_sd.pop(fp, None)
            if removed is None:
                return
            self._ask_depth -= removed
            if fp == self._best_ask:
                self._best_ask = self._asks_sd.peekitem(0)[0] if self._asks_sd else None
        self._asks_cache = None

    # ── best-bid-ask replacement (for best_bid_ask WebSocket events) ──────────

    def replace_best_bid(self, price: Decimal, size: Decimal | None = None) -> None:
        """Update top-of-book bid from a best_bid_ask WS event.

        Fixes two problems present in the naive implementation:

        1. **Fake liquidity**: the old code carried the old top-of-book size to
           the new price.  A best_bid_ask event carries no depth information,
           so that carried size was invented.  We now use 0.0 for an unknown
           size at a new price, which correctly signals "no known depth" to the
           scanner's depth filter.

        2. **O(n) stale prune**: the old code built a list comprehension
           ``[k for k in sd if k > fp]`` which is O(n).  This version uses
           ``peekitem(0)`` (O(log n)) and pops from the front of the SortedDict
           in a tight loop, which is O(k) where k is the number of stale levels
           (almost always 0 or 1).
        """
        sd = self._bids_sd
        fp = float(price)

        # ── Prune stale levels above new best bid (O(k)) ──────────────────────
        # SortedDict(lambda p: -p): peekitem(0) = highest price (best bid).
        # A stale bid has price > fp — pop from the front until clean.
        while sd:
            top_price, top_size = sd.peekitem(0)
            if top_price <= fp:
                break
            sd.popitem(0)
            self._bid_depth -= top_size

        # ── Apply new level ────────────────────────────────────────────────────
        existing = sd.get(fp)
        if size is not None:
            effective_size = float(size)           # caller supplied size
        elif existing is not None:
            effective_size = existing              # price unchanged: keep real size
        else:
            effective_size = 0.0                   # new price from BBA: size unknown
        old_at_fp = existing if existing is not None else 0.0
        self._bid_depth += effective_size - old_at_fp
        sd[fp] = effective_size
        self._best_bid = fp
        self._bids_cache = None

    def replace_best_ask(self, price: Decimal, size: Decimal | None = None) -> None:
        """Update top-of-book ask from a best_bid_ask WS event.

        Mirrors ``replace_best_bid``.  Fixes the same fake-liquidity and O(n)
        stale-prune issues on the ask side.
        """
        sd = self._asks_sd
        fp = float(price)

        # ── Prune stale levels below new best ask (O(k)) ──────────────────────
        # SortedDict (ascending): peekitem(0) = lowest price (best ask).
        # A stale ask has price < fp — pop from the front until clean.
        while sd:
            bot_price, bot_size = sd.peekitem(0)
            if bot_price >= fp:
                break
            sd.popitem(0)
            self._ask_depth -= bot_size

        # ── Apply new level ────────────────────────────────────────────────────
        existing = sd.get(fp)
        if size is not None:
            effective_size = float(size)
        elif existing is not None:
            effective_size = existing
        else:
            effective_size = 0.0
        old_at_fp = existing if existing is not None else 0.0
        self._ask_depth += effective_size - old_at_fp
        sd[fp] = effective_size
        self._best_ask = fp
        self._asks_cache = None

    # ── Derived book statistics — O(1) reads from cached scalars ─────────────

    @property
    def best_bid(self) -> float | None:
        """Best bid price as float. O(1) — returns cached scalar."""
        return self._best_bid

    @property
    def best_ask(self) -> float | None:
        """Best ask price as float. O(1) — returns cached scalar."""
        return self._best_ask

    @property
    def spread(self) -> float | None:
        bb, ba = self._best_bid, self._best_ask
        return (ba - bb) if bb is not None and ba is not None else None

    @property
    def ask_depth(self) -> float:
        """Total ask-side depth. O(1) — returns cached scalar."""
        return self._ask_depth

    @property
    def bid_depth(self) -> float:
        """Total bid-side depth. O(1) — returns cached scalar."""
        return self._bid_depth

    def avg_ask_price_for_size(self, size: Decimal) -> float | None:
        """VWAP ask price to fill ``size`` shares. Returns None if depth insufficient."""
        fs = float(size)
        if fs <= 0.0:
            return None
        # Fast early-exit: total depth check before walking levels.
        if self._ask_depth < fs:
            return None
        remaining, total, filled = fs, 0.0, 0.0
        for price, level_size in self._asks_sd.items():
            take = min(level_size, remaining)
            if take <= 0.0:
                continue
            total += take * price
            filled += take
            remaining -= take
            if remaining <= 0.0:
                break
        return (total / fs) if filled >= fs else None

    def avg_bid_price_for_size(self, size: Decimal) -> float | None:
        """VWAP bid price to fill ``size`` shares. Returns None if depth insufficient."""
        fs = float(size)
        if fs <= 0.0:
            return None
        if self._bid_depth < fs:
            return None
        remaining, total, filled = fs, 0.0, 0.0
        for price, level_size in self._bids_sd.items():
            take = min(level_size, remaining)
            if take <= 0.0:
                continue
            total += take * price
            filled += take
            remaining -= take
            if remaining <= 0.0:
                break
        return (total / fs) if filled >= fs else None

    @classmethod
    def from_api(cls, payload: dict[str, Any], token_id: str | None = None) -> "OrderBook":
        asset_id = str(payload.get("asset_id") or payload.get("assetId") or token_id or "")
        bids = [OrderBookLevel.from_api(item) for item in payload.get("bids") or []]
        asks = [OrderBookLevel.from_api(item) for item in payload.get("asks") or []]
        return cls(
            asset_id=asset_id,
            bids=bids,
            asks=asks,
            timestamp=_coerce_timestamp(payload.get("timestamp")),
            raw_json=payload,
        )

    def __repr__(self) -> str:
        return f"OrderBook(asset_id={self.asset_id!r}, bids={len(self._bids_sd)}, asks={len(self._asks_sd)})"


def _avg_price_for_size(levels: list[OrderBookLevel], size: Decimal) -> Decimal | None:
    """Kept for any code that still calls it with a plain list."""
    if size <= ZERO:
        return None
    remaining, total, filled = size, ZERO, ZERO
    for level in levels:
        take = min(level.size, remaining)
        if take <= ZERO:
            continue
        total += take * level.price
        filled += take
        remaining -= take
        if remaining <= ZERO:
            break
    return (total / size) if filled >= size else None


def _coerce_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        numeric = int(str(value))
    except (TypeError, ValueError):
        return None
    seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
    return datetime.fromtimestamp(seconds, tz=UTC)
