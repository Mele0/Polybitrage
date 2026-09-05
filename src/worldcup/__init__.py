"""World Cup arbitrage subsystem.

Discovers Polymarket World Cup events/markets, infers logical relations between
them, and scans the public CLOB order books for deterministic arbitrage and
statistical edges.  It reuses the existing order-book models, CLOB client,
WebSocket cache, depth-aware sizing math, and guarded live executor rather than
reimplementing them.

Modules:
    schema           normalized market schema + market-type classifier
    discovery        Gamma-API World Cup event/market discovery + caching
    relations        relation inference engine (subset / complement / quantity)
    state_validator  per-state worst-case payout validator
    scanner          orchestration: discovery -> relations -> executable scan
    executor         exposure limiter / kill switch wrapping the live trader
"""
