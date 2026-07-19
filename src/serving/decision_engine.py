"""Cost-based decision: turns a fraud probability into allow/review/block.

Uses the standard cost-sensitive-learning threshold (Elkan, "The Foundations
of Cost-Sensitive Learning", 2001): for a binary cost matrix, the
Bayes-optimal probability cutoff for predicting the positive class is

    p* = C_FP / (C_FP + C_FN)

where C_FP is the cost of a false positive (wrongly blocking a legitimate
transaction) and C_FN is the cost of a false negative (missing real fraud).
We use the transaction's own `amount` as C_FN — the actual dollar loss if
this specific transaction is fraud and isn't stopped. That makes the
threshold amount-dependent on purpose: a five-dollar transaction almost
never justifies blocking, an $800,000 one needs very little suspicion to.

`false_positive_cost` (configs/decision.yaml) is a placeholder, not a real
business figure — see docs/DECISIONS.md. Nothing else here needs to change
once a real one is supplied.
"""

from __future__ import annotations

from src.config import DecisionSettings

_settings = DecisionSettings()  # type: ignore[call-arg]  # fields load from configs/decision.yaml


def block_threshold(amount: float, false_positive_cost: float = _settings.false_positive_cost) -> float:
    """Bayes-optimal probability cutoff for blocking a transaction of this amount."""
    return false_positive_cost / (false_positive_cost + amount)


def decide(fraud_probability: float, amount: float) -> str:
    """allow / review / block, from the cost-optimal threshold for this transaction's amount."""
    block_at = block_threshold(amount)
    review_at = block_at * _settings.review_band_fraction

    if fraud_probability >= block_at:
        return "block"
    if fraud_probability >= review_at:
        return "review"
    return "allow"
