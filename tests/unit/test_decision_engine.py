from src.serving.decision_engine import block_threshold, decide


def test_larger_amount_needs_less_suspicion_to_block() -> None:
    small = block_threshold(amount=5.0)
    large = block_threshold(amount=800_000.0)

    assert large < small


def test_decide_blocks_above_threshold() -> None:
    threshold = block_threshold(amount=100.0)

    assert decide(fraud_probability=threshold, amount=100.0) == "block"
    assert decide(fraud_probability=1.0, amount=100.0) == "block"


def test_decide_reviews_in_the_band_below_threshold() -> None:
    threshold = block_threshold(amount=100.0)

    assert decide(fraud_probability=threshold * 0.75, amount=100.0) == "review"


def test_decide_allows_low_probability() -> None:
    assert decide(fraud_probability=0.0, amount=100.0) == "allow"
