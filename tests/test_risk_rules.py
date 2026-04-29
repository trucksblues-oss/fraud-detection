import pytest
from risk_rules import label_risk, score_transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A baseline transaction that scores 10 (only the mid-range amount rule fires).
# Used to test individual signals in isolation by overriding one field at a time.
BASE_TX = {
    "device_risk_score": 20,   # < 40, no device contribution
    "is_international": 0,
    "amount_usd": 600,         # 500–999 → +10
    "velocity_24h": 2,         # < 3, no velocity contribution
    "failed_logins_24h": 0,    # < 2, no login contribution
    "prior_chargebacks": 0,
}


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_low():
    assert label_risk(10) == "low"

def test_label_risk_medium():
    assert label_risk(35) == "medium"

def test_label_risk_high():
    assert label_risk(75) == "high"

def test_label_risk_boundary_low_to_medium():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"

def test_label_risk_boundary_medium_to_high():
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"

def test_label_risk_zero_is_low():
    assert label_risk(0) == "low"

def test_label_risk_100_is_high():
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — amount and login signals (correct in current code)
# ---------------------------------------------------------------------------

def test_large_amount_adds_risk():
    tx = {**BASE_TX, "amount_usd": 1200}
    assert score_transaction(tx) >= 25

def test_mid_range_amount_adds_risk():
    tx = {**BASE_TX, "amount_usd": 700}
    assert score_transaction(tx) > score_transaction({**BASE_TX, "amount_usd": 50})

def test_failed_logins_high_adds_risk():
    tx = {**BASE_TX, "failed_logins_24h": 5}
    assert score_transaction(tx) > score_transaction(BASE_TX)

def test_failed_logins_medium_adds_risk():
    tx = {**BASE_TX, "failed_logins_24h": 3}
    assert score_transaction(tx) > score_transaction(BASE_TX)

def test_medium_device_risk_adds_risk():
    # device_risk 40–69 should add points (this rule is correct in current code)
    tx = {**BASE_TX, "device_risk_score": 50}
    assert score_transaction(tx) > score_transaction(BASE_TX)

def test_medium_velocity_adds_risk():
    # velocity 3–5 should add points (this rule is correct in current code)
    tx = {**BASE_TX, "velocity_24h": 4}
    assert score_transaction(tx) > score_transaction(BASE_TX)


# ---------------------------------------------------------------------------
# score_transaction — score bounds
# ---------------------------------------------------------------------------

def test_score_never_below_zero():
    tx = {**BASE_TX, "amount_usd": 5}   # no positive rules fire at all
    assert score_transaction(tx) >= 0

def test_score_never_above_100():
    tx = {
        "device_risk_score": 90,
        "is_international": 1,
        "amount_usd": 2000,
        "velocity_24h": 10,
        "failed_logins_24h": 10,
        "prior_chargebacks": 5,
    }
    assert score_transaction(tx) <= 100


# ---------------------------------------------------------------------------
# score_transaction — the four known bugs (these tests define correct behavior
# and will FAIL until the scoring logic is fixed)
# ---------------------------------------------------------------------------

def test_high_device_risk_increases_score():
    # device_risk >= 70 signals a compromised or emulated device and must add risk
    high_device = {**BASE_TX, "device_risk_score": 75}
    assert score_transaction(high_device) > score_transaction(BASE_TX)

def test_international_transaction_increases_score():
    # cross-border transactions carry higher fraud risk and must add risk
    intl = {**BASE_TX, "is_international": 1}
    assert score_transaction(intl) > score_transaction(BASE_TX)

def test_high_velocity_increases_score():
    # >= 6 transactions in 24h is a card-testing or rapid-fraud pattern and must add risk
    high_vel = {**BASE_TX, "velocity_24h": 7}
    assert score_transaction(high_vel) > score_transaction(BASE_TX)

def test_prior_chargebacks_increase_score():
    # a history of chargebacks is the strongest predictor of future fraud
    repeat = {**BASE_TX, "prior_chargebacks": 2}
    assert score_transaction(repeat) > score_transaction(BASE_TX)

def test_single_prior_chargeback_increases_score():
    one_cb = {**BASE_TX, "prior_chargebacks": 1}
    assert score_transaction(one_cb) > score_transaction(BASE_TX)
