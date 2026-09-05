"""Book state snapshot: atomic JSON serialisation of WS order-book state.

On a clean restart, loading a fresh snapshot eliminates the ~10-second
cold-start window where every book is ``None`` and all REST fallbacks must
fire before the scanner can score any pair.

Format (JSON, written atomically via temp-file + rename):
::

    {
      "saved_at": 1718000000.123,     # wall-clock epoch seconds
      "version": 1,
      "books": {
        "<token_id>": {
          "bids": [[price, size], ...],  # sorted descending — float pairs
          "asks": [[price, size], ...]   # sorted ascending — float pairs
        }
      }
    }

The snapshot is rejected on load if it is older than ``max_age_seconds``
(default 120 s), so stale snapshots from earlier sessions do no harm.

No external dependencies beyond the stdlib.  Failures are logged at WARNING
level and never propagate — snapshot loading is strictly best-effort.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.polymarket.models import OrderBook

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


# ── Serialisation ─────────────────────────────────────────────────────────────

def _book_to_dict(book: "OrderBook") -> dict[str, Any]:
    """Serialise an OrderBook's float-keyed SortedDicts to a plain dict."""
    return {
        "bids": [[p, s] for p, s in book._bids_sd.items()],
        "asks": [[p, s] for p, s in book._asks_sd.items()],
    }


def _dict_to_book(data: dict[str, Any], token_id: str) -> "OrderBook | None":
    """Reconstruct an OrderBook from a plain dict.  Returns None on error."""
    try:
        from src.polymarket.models import OrderBook, OrderBookLevel
        from decimal import Decimal

        bids = [
            OrderBookLevel(price=Decimal(str(p)), size=Decimal(str(s)))
            for p, s in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(p)), size=Decimal(str(s)))
            for p, s in data.get("asks", [])
        ]
        return OrderBook(asset_id=token_id, bids=bids, asks=asks)
    except Exception as exc:
        logger.warning("book_snapshot: failed to reconstruct book for %s: %s", token_id, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def save_snapshot(
    states: dict[str, Any],  # dict[str, WebSocketOrderBookState]
    path: Path,
) -> None:
    """Atomically write current book states to *path* as JSON.

    *states* is ``ClobWebSocketClient._states``.  The write is atomic (temp
    file + ``os.replace``) so a concurrent reader never sees a partial file.
    Errors are logged at WARNING level and swallowed — callers must not rely
    on this succeeding.
    """
    try:
        books: dict[str, Any] = {}
        for token_id, state in states.items():
            if state.book is not None:
                try:
                    books[token_id] = _book_to_dict(state.book)
                except Exception as exc:
                    logger.debug("book_snapshot: skipping token %s: %s", token_id, exc)

        payload = {
            "saved_at": time.time(),
            "version": _FORMAT_VERSION,
            "books": books,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".book_snapshot_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.debug("book_snapshot: saved %d books to %s", len(books), path)
    except Exception as exc:
        logger.warning("book_snapshot: save failed: %s", exc)


def load_snapshot(
    path: Path,
    *,
    max_age_seconds: float = 120.0,
) -> dict[str, "OrderBook"] | None:
    """Load a previously saved snapshot from *path*.

    Returns ``dict[token_id, OrderBook]`` ready to pass to
    ``ClobWebSocketClient.seed_books()``.  Returns ``None`` when:

    * the file does not exist
    * the file is older than *max_age_seconds*
    * any JSON or schema error occurs

    Errors are logged at WARNING level and never propagate.
    """
    try:
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        version = payload.get("version", 0)
        if version != _FORMAT_VERSION:
            logger.warning(
                "book_snapshot: unsupported version %s in %s, skipping", version, path
            )
            return None

        saved_at = float(payload.get("saved_at", 0.0))
        age = time.time() - saved_at
        if age > max_age_seconds:
            logger.info(
                "book_snapshot: snapshot at %s is %.0f s old (max %s s), skipping",
                path, age, max_age_seconds,
            )
            return None

        raw_books: dict[str, Any] = payload.get("books", {})
        books: dict[str, "OrderBook"] = {}
        for token_id, data in raw_books.items():
            book = _dict_to_book(data, token_id)
            if book is not None:
                books[token_id] = book

        logger.info(
            "book_snapshot: loaded %d/%d books from %s (%.1f s old)",
            len(books), len(raw_books), path, age,
        )
        return books or None  # return None rather than empty dict

    except Exception as exc:
        logger.warning("book_snapshot: load failed from %s: %s", path, exc)
        return None
