"""CLI runners for the World Cup subsystem, dispatched from src.main."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from src.worldcup.discovery import discover_world_cup, save_discovery_cache
from src.worldcup.relations import infer_relations, relation_to_pair_dict
from src.worldcup.scanner import WorldCupScanner, WorldCupScannerConfig


def _load_config(args: argparse.Namespace) -> WorldCupScannerConfig:
    cfg = WorldCupScannerConfig.from_yaml(Path(args.config)) if args.config and Path(args.config).exists() else WorldCupScannerConfig()
    # CLI overrides (only when explicitly provided)
    if getattr(args, "execution_mode", None):
        cfg.execution_mode = args.execution_mode
        cfg.paper_trade = args.execution_mode != "live"
    for name in ("min_edge_bps", "max_leg_slippage", "fee_rate", "min_liquidity", "max_stale_ms",
                 "max_trade_size", "max_event_exposure", "show_top"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if getattr(args, "enable_statistical", False):
        cfg.enable_statistical_arbs = True
    if getattr(args, "no_deterministic", False):
        cfg.enable_deterministic_arbs = False
    if getattr(args, "out_dir", None):
        cfg.out_dir = Path(args.out_dir)
    return cfg


async def run_worldcup_discover(args: argparse.Namespace) -> None:
    result = await discover_world_cup(
        include_closed=args.include_closed,
        include_futures=not args.no_futures,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_discovery_cache(result, out_dir / "discovery_cache.json")
    graph = infer_relations(
        result,
        overrides_path=(Path(args.overrides) if args.overrides and Path(args.overrides).exists() else None),
    )
    pairs = [relation_to_pair_dict(r, enabled=False) for r in graph.subset_relations]
    (out_dir / "generated_worldcup_pairs.yaml").write_text(
        yaml.safe_dump({"notes": "Auto-discovered World Cup subset arbs. Paper-only.",
                        "pairs": pairs}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out_dir / "relations.json").write_text(json.dumps([
        {"name": r.name, "type": r.relation_type, "subtype": r.relation_subtype,
         "payout": r.guaranteed_payout, "safety": r.safety, "teams": list(r.teams),
         "legs": [{"token_id": l.token_id, "side": l.side, "label": l.label} for l in r.legs]}
        for r in graph.relations], indent=2), encoding="utf-8")

    print("World Cup discovery complete (paper-only; generated pairs disabled by default).")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")
    print(f"  relations: {graph.summary()}")
    print(f"  subset-arb watchlist: {out_dir / 'generated_worldcup_pairs.yaml'} ({len(pairs)} pairs)")
    print(f"  relations dump:       {out_dir / 'relations.json'}")
    print("\nSample games discovered:")
    for g in result.all_games[:12]:
        kinds = sorted({m.kind.value for m in g.markets})
        print(f"  - {g.label:<28} game_id={g.game_id} markets={len(g.markets)} kinds={kinds}")


async def run_worldcup_scan(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    live_trader = None
    if cfg.execution_mode == "live":
        from src.polymarket.live_trader import LiveTradingConfig, LiveTrader
        live_cfg = LiveTradingConfig.from_env(
            max_session_spend=args.live_max_session_spend or cfg.max_total_exposure,
            max_legs_per_bundle=args.live_max_legs_per_bundle,
            confirmation=args.live_confirmation,
            compliance_ack=args.live_compliance_ack,
        )
        live_trader = LiveTrader(live_cfg)

    scanner = WorldCupScanner(cfg, live_trader=live_trader)
    execute = not args.no_execute
    deadline = None if args.duration_minutes is None else (asyncio.get_event_loop().time() + args.duration_minutes * 60)
    use_cache = False  # first scan always refreshes discovery
    try:
        while True:
            report = await scanner.scan_once(use_cache=use_cache, execute=execute)
            use_cache = True
            print(scanner.render_table(report))
            if report.actions:
                print("actions:")
                for a in report.actions:
                    print(f"  - {a}")
            if args.once:
                break
            if deadline is not None and asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(max(0.5, args.poll_seconds))
    finally:
        await scanner.book_source.stop()
