# Security Policy

Polybitrage can connect to real trading venues and, in live mode, handle wallet
keys and API credentials. Please treat security issues with care.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, report it privately using GitHub's private vulnerability reporting: on
the repository's **Security** tab, choose **Report a vulnerability**.

Please include a description of the issue, steps to reproduce, the potential
impact, and any suggested remediation. You can expect an acknowledgement within a
few days. Please give a reasonable window to address the issue before any public
disclosure.

## Handling credentials and keys

- **Never** include private keys, API keys, passphrases, wallet addresses, or the
  contents of a `.env` file in an issue, pull request, log paste, or commit.
- `.env` and generated `reports/` output are gitignored; keep real credentials
  local and out of version control.
- If you believe a credential has been exposed in this repository or its history,
  rotate it immediately and report it privately using the channel above.

## Scope

Because live trading moves real money, reports that concern the live-execution
path (order signing, submission, risk limits, or the process lock) are treated
with the highest priority.
