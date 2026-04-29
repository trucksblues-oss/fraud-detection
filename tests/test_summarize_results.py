import pandas as pd
import pytest
from analyze_fraud import summarize_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scored(*rows):
    """(transaction_id, amount_usd, risk_label)"""
    return pd.DataFrame(rows, columns=["transaction_id", "amount_usd", "risk_label"])


def make_chargebacks(*rows):
    """(transaction_id, loss_amount_usd)"""
    return pd.DataFrame(rows, columns=["transaction_id", "loss_amount_usd"])


def row(summary, label):
    return summary[summary["risk_label"].astype(str) == label].iloc[0]


# ---------------------------------------------------------------------------
# Output shape and columns
# ---------------------------------------------------------------------------

def test_output_has_required_columns():
    scored = make_scored((1, 100.0, "low"))
    cb = make_chargebacks()
    summary = summarize_results(scored, cb)
    required = {"risk_label", "transactions", "total_amount_usd", "avg_amount_usd",
                "chargebacks", "confirmed_loss_usd", "chargeback_rate"}
    assert required.issubset(set(summary.columns))


def test_one_row_per_risk_label():
    scored = make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "low"),
        (3, 500.0, "medium"),
        (4, 1000.0, "high"),
    )
    summary = summarize_results(scored, make_chargebacks())
    assert len(summary) == 3


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------

def test_sort_order_is_low_medium_high():
    scored = make_scored(
        (1, 100.0, "high"),
        (2, 200.0, "low"),
        (3, 300.0, "medium"),
    )
    summary = summarize_results(scored, make_chargebacks())
    assert list(summary["risk_label"].astype(str)) == ["low", "medium", "high"]


def test_sort_order_with_only_two_labels():
    scored = make_scored((1, 100.0, "high"), (2, 200.0, "low"))
    summary = summarize_results(scored, make_chargebacks())
    assert list(summary["risk_label"].astype(str)) == ["low", "high"]


# ---------------------------------------------------------------------------
# Transaction counts and amounts
# ---------------------------------------------------------------------------

def test_transaction_counts_per_label():
    scored = make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "low"),
        (3, 500.0, "medium"),
        (4, 1000.0, "high"),
    )
    summary = summarize_results(scored, make_chargebacks())
    assert row(summary, "low")["transactions"] == 2
    assert row(summary, "medium")["transactions"] == 1
    assert row(summary, "high")["transactions"] == 1


def test_total_amount_per_label():
    scored = make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "low"),
        (3, 500.0, "high"),
    )
    summary = summarize_results(scored, make_chargebacks())
    assert row(summary, "low")["total_amount_usd"] == pytest.approx(300.0)
    assert row(summary, "high")["total_amount_usd"] == pytest.approx(500.0)


def test_avg_amount_per_label():
    scored = make_scored(
        (1, 100.0, "low"),
        (2, 300.0, "low"),
        (3, 800.0, "high"),
    )
    summary = summarize_results(scored, make_chargebacks())
    assert row(summary, "low")["avg_amount_usd"] == pytest.approx(200.0)
    assert row(summary, "high")["avg_amount_usd"] == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# Chargeback counts
# ---------------------------------------------------------------------------

def test_chargeback_count_per_label():
    scored = make_scored(
        (1, 100.0, "low"),
        (2, 200.0, "high"),
        (3, 300.0, "high"),
    )
    cb = make_chargebacks((2, 180.0), (3, 270.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "low")["chargebacks"] == 0
    assert row(summary, "high")["chargebacks"] == 2


def test_label_with_no_chargebacks_shows_zero():
    scored = make_scored((1, 100.0, "low"), (2, 200.0, "medium"))
    summary = summarize_results(scored, make_chargebacks())
    assert row(summary, "low")["chargebacks"] == 0
    assert row(summary, "medium")["chargebacks"] == 0


def test_chargeback_not_in_scored_is_ignored():
    # transaction 99 is in chargebacks but was never scored — must not inflate counts
    scored = make_scored((1, 100.0, "low"))
    cb = make_chargebacks((1, 100.0), (99, 999.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "low")["chargebacks"] == 1


# ---------------------------------------------------------------------------
# Confirmed loss dollars
# ---------------------------------------------------------------------------

def test_confirmed_loss_usd_per_label():
    scored = make_scored(
        (1, 500.0, "high"),
        (2, 300.0, "high"),
        (3, 100.0, "low"),
    )
    cb = make_chargebacks((1, 450.0), (2, 270.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "high")["confirmed_loss_usd"] == pytest.approx(720.0)
    assert row(summary, "low")["confirmed_loss_usd"] == pytest.approx(0.0)


def test_confirmed_loss_usd_excludes_unconfirmed_transactions():
    # transaction 2 was never charged back — its amount must not appear in confirmed losses
    scored = make_scored((1, 500.0, "high"), (2, 300.0, "high"))
    cb = make_chargebacks((1, 450.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "high")["confirmed_loss_usd"] == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# Chargeback rate
# ---------------------------------------------------------------------------

def test_chargeback_rate_calculation():
    scored = make_scored(
        (1, 100.0, "high"),
        (2, 200.0, "high"),
        (3, 300.0, "high"),
        (4, 400.0, "high"),
    )
    cb = make_chargebacks((1, 100.0), (2, 200.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "high")["chargeback_rate"] == pytest.approx(0.5)


def test_chargeback_rate_zero_when_no_chargebacks():
    scored = make_scored((1, 100.0, "low"), (2, 200.0, "low"))
    summary = summarize_results(scored, make_chargebacks())
    assert row(summary, "low")["chargeback_rate"] == pytest.approx(0.0)


def test_chargeback_rate_one_when_all_charged_back():
    scored = make_scored((1, 100.0, "high"), (2, 200.0, "high"))
    cb = make_chargebacks((1, 100.0), (2, 200.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "high")["chargeback_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Duplicate chargeback rows (data integrity)
# ---------------------------------------------------------------------------

def test_duplicate_chargebacks_not_double_counted():
    # same transaction_id appears twice in chargebacks (re-filed dispute / data error)
    scored = make_scored((1, 500.0, "high"), (2, 200.0, "low"))
    cb = make_chargebacks((1, 500.0), (1, 500.0))
    summary = summarize_results(scored, cb)
    assert row(summary, "high")["chargebacks"] == 1
    assert row(summary, "high")["confirmed_loss_usd"] == pytest.approx(500.0)
    assert row(summary, "high")["chargeback_rate"] == pytest.approx(1.0)
