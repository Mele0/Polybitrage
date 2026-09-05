"""State-space validator for World Cup arbitrage baskets.

Given the legs of a proposed trade (each leg buys a specific YES/NO token) and an
executable per-unit cost, enumerate every relevant terminal outcome state and
confirm the minimum payout strictly exceeds cost after a safety buffer.  A trade
is only allowed when the worst case is profitable — this is the guard that turns
an inferred *relation* into an executable *arbitrage*.

A state is represented as ``{token_id: payout_per_share_if_this_state_occurs}``.
For a leg that bought token T, its payout in a state is ``state.get(T, 0.0)``
(1.0 when T resolves YES, else 0.0).  ``synthesize_states`` derives the states
for the two structural primitives (subset, exhaustive partition); explicit states
can also be supplied for fragile baskets such as exact-score combinations.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.worldcup.relations import COMPLEMENT, QUANTITY, SUBSET, Relation, RelationLeg


@dataclass
class ValidationResult:
    ok: bool
    min_payout: float
    cost: float
    worst_case_profit: float       # min_payout - cost (no buffer)
    net_worst_case_profit: float   # min_payout - cost - buffer
    n_states: int
    reason: str

    def as_dict(self) -> dict[str, float | bool | int | str]:
        return {
            "ok": self.ok,
            "min_payout": round(self.min_payout, 6),
            "cost": round(self.cost, 6),
            "worst_case_profit": round(self.worst_case_profit, 6),
            "net_worst_case_profit": round(self.net_worst_case_profit, 6),
            "n_states": self.n_states,
            "reason": self.reason,
        }


def validate_legs(
    legs: tuple[RelationLeg, ...] | list[RelationLeg],
    states: tuple[dict[str, float], ...] | list[dict[str, float]],
    *,
    cost: float,
    buffer: float = 0.0,
) -> ValidationResult:
    """Validate a basket against an explicit list of states."""
    if not states:
        return ValidationResult(False, 0.0, cost, -cost, -cost - buffer, 0, "no states to validate")
    payouts: list[float] = []
    for state in states:
        payout = 0.0
        for leg in legs:
            payout += float(state.get(leg.token_id, 0.0))
        payouts.append(payout)
    min_payout = min(payouts)
    wcp = min_payout - cost
    net = wcp - buffer
    ok = net > 0.0
    reason = "worst-case state profitable" if ok else "worst-case state not profitable after buffer"
    return ValidationResult(
        ok=ok,
        min_payout=min_payout,
        cost=cost,
        worst_case_profit=wcp,
        net_worst_case_profit=net,
        n_states=len(states),
        reason=reason,
    )


def synthesize_states(relation: Relation) -> tuple[dict[str, float], ...]:
    """Build the terminal states for a relation when none are supplied.

    * ``subset`` (parent YES + child NO): three states — child true, child false &
      parent true, parent false — with payouts {1, 2, 1}.
    * ``complement_basket`` / ``quantity_partition``: one winning state per leg
      (exactly one leg resolves to its bought side), payout 1 in each.
    """
    if relation.states:
        return relation.states
    if relation.relation_type == SUBSET and len(relation.legs) == 2:
        parent, child = relation.legs  # parent bought YES, child bought NO
        # child YES => parent YES too
        s1 = {parent.token_id: 1.0, child.token_id: 0.0}
        # parent YES, child false (we hold child NO -> pays)
        s2 = {parent.token_id: 1.0, child.token_id: 1.0}
        # parent NO (=> child NO too, child NO pays)
        s3 = {parent.token_id: 0.0, child.token_id: 1.0}
        return (s1, s2, s3)
    if relation.relation_type in (COMPLEMENT, QUANTITY):
        states: list[dict[str, float]] = []
        for winner in relation.legs:
            states.append({leg.token_id: (1.0 if leg.token_id == winner.token_id else 0.0) for leg in relation.legs})
        return tuple(states)
    return ()


def validate_relation(relation: Relation, *, cost: float, buffer: float = 0.0) -> ValidationResult:
    """Validate a relation at the given executable unit cost."""
    states = synthesize_states(relation)
    return validate_legs(relation.legs, states, cost=cost, buffer=buffer)


# ── Helpers for constructing explicit exact-score / BTTS basket states ──────────


def build_btts_exact_basket_states(
    btts_yes_token: str,
    exact_no_tokens: dict[str, tuple[int, int]],
    *,
    extra_btts_scores: tuple[tuple[int, int], ...] = ((2, 1), (1, 2), (3, 1)),
    no_btts_scores: tuple[tuple[int, int], ...] = ((1, 0), (0, 0), (2, 0)),
) -> tuple[dict[str, float], ...]:
    """States for: buy BTTS-Yes + buy NO on a set of exact scores.

    Legs pay: BTTS-Yes token pays 1 when both teams score; each exact NO token
    pays 1 unless that exact score occurs.  We enumerate: each named exact score,
    additional both-scored scores, and no-both-scored scores.
    """
    states: list[dict[str, float]] = []
    all_exacts = list(exact_no_tokens.items())

    def make_state(score: tuple[int, int]) -> dict[str, float]:
        a, b = score
        both = a > 0 and b > 0
        st: dict[str, float] = {btts_yes_token: 1.0 if both else 0.0}
        for token, exact_score in all_exacts:
            st[token] = 0.0 if exact_score == score else 1.0
        return st

    for _, score in all_exacts:
        states.append(make_state(score))
    for score in extra_btts_scores:
        states.append(make_state(score))
    for score in no_btts_scores:
        states.append(make_state(score))
    return tuple(states)
