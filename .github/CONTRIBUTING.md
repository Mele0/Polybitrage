# Contributing to Polybitrage

Thank you for your interest in contributing. Polybitrage is a high-frequency
trading engine for prediction markets, so contributions are held to two extra
standards on top of the usual ones: the hot path must stay fast, and anything
touching live execution must stay safe. This guide explains how to get set up and
what a good contribution looks like.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **New venue adapters** or order-book providers beyond Polymarket CLOB and Kalshi
- **Arbitrage relations and strategies** — additional N-leg packages, sizing, and exit logic
- **Latency optimizations** on the hot path — the Rust WebSocket client, the Cython scan kernel, or the async scan loop
- **Live-execution safety** — risk controls, order handling, reconciliation
- **Observability** — new Grafana panels or Prometheus metrics
- **Market discovery** — new candidate sources and pair-generation heuristics
- **Documentation, tests, and reproducibility improvements**
- **Bug fixes**

## Getting started

1. Fork the repository and clone your fork.
2. Create a branch for your change:

   ```bash
   git checkout -b feature/short-description
   ```

3. Create a virtual environment and install the project with dev dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev]'
   ```

4. (Optional) Build the native fast paths if your change touches them. Both are
   optional — the bot falls back to pure Python if they are absent.

   ```bash
   python -m pip install -e '.[speed]'
   python setup_cython.py build_ext --inplace     # C / Cython scan kernel
   cd rust_ws_client && cargo build --release && cd ..   # Rust WebSocket client
   ```

5. Make your changes, add or update tests, and run the suite (see below).
6. Push your branch and open a pull request.

## Running tests

```bash
python -m pytest
```

Please add tests for new behaviour and make sure the existing suite passes before
opening a pull request. Changes to sizing, relation classification, or the audit
path should come with a test that covers the new case.

## Coding guidelines

- **Match the surrounding code.** Follow the style, naming, and type-hint
  conventions already in the module you are editing.
- **Keep the hot path lean.** The scan loop and the Rust/Cython kernels are
  latency-sensitive. Avoid per-scan allocations, avoid blocking calls on the
  asyncio event loop, and prefer the existing numpy/compiled paths over new
  pure-Python loops. If a change affects scan latency, include before/after
  numbers from `poly-arb benchmark-scan`.
- **Default to paper mode.** New features should work and be testable in paper
  mode without credentials.
- **Never commit secrets.** No private keys, API keys, wallet addresses, `.env`
  files, or generated `reports/` output. `.env.example` documents the variables;
  real values stay local.

## Reporting bugs

Open an issue using the bug report template and include:

- A clear description of the problem
- Steps to reproduce it
- Expected behaviour
- Actual behaviour
- Your environment (OS, Python version, relevant package versions, and whether
  the Rust/Cython extensions were built)

Do not paste credentials, private keys, or full audit logs into an issue.

## Proposing features and larger changes

For substantial changes — new venues, new strategy types, and especially anything
that touches the live-execution path — please open an issue to discuss the design
before implementing it. This avoids wasted work and keeps the execution and risk
model coherent.

## Pull request checklist

- [ ] The change is focused and described clearly in the PR
- [ ] Tests were added or updated, and `pytest` passes
- [ ] No secrets, credentials, or generated output are included
- [ ] Hot-path changes include benchmark numbers where relevant
- [ ] Live-execution changes were discussed in an issue first
