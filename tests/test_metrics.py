from __future__ import annotations

import pandas as pd

from promolab.metrics import compute_kpis, compute_lift, compute_lift_for_window, get_baseline_window


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T10:00:00Z",
                "2025-01-02T10:00:00Z",
                "2025-01-08T10:00:00Z",
                "2025-01-09T10:00:00Z",
            ],
            "order_id": ["o1", "o2", "o3", "o4"],
            "item_name": ["A", "B", "A", "B"],
            "quantity": [1, 2, 1, 2],
            "unit_price": [10.0, 10.0, 20.0, 20.0],
            "discount_amount": [0.0, 1.0, 0.0, 2.0],
            "refund_amount": [0.0, 0.0, 0.0, 1.0],
            "cogs_amount": [3.0, 5.0, 6.0, 8.0],
        }
    )


def test_compute_kpis_exact() -> None:
    k = compute_kpis(_df(), "2025-01-01T00:00:00Z", "2025-01-02T23:59:59Z")
    assert k["gross_sales"] == 30.0
    assert k["revenue"] == 29.0
    assert k["orders"] == 2.0
    assert k["aov"] == 14.5
    assert round(k["discount_rate"], 6) == round(1.0 / 30.0, 6)
    assert k["refund_rate"] == 0.0
    assert k["gross_profit"] == 21.0


def test_get_baseline_window_matched_4w() -> None:
    b = get_baseline_window(_df(), "2025-01-08T00:00:00Z", "2025-01-09T23:59:59Z", method="matched_4w")
    assert b["type"] == "windows"
    assert len(b["windows"]) == 4


def test_get_baseline_window_last_28d() -> None:
    b = get_baseline_window(_df(), "2025-01-08T00:00:00Z", "2025-01-09T23:59:59Z", method="last_28d")
    assert b["type"] == "range"
    assert str(b["start"]) == "2024-12-11 00:00:00+00:00"
    assert str(b["end"]) == "2025-01-07 23:59:59+00:00"


def test_compute_lift_math() -> None:
    promo = {"revenue": 120.0, "orders": 10.0}
    base = {"revenue": 100.0, "orders": 8.0}
    lift = compute_lift(promo, base)
    assert lift["revenue"]["abs_change"] == 20.0
    assert lift["revenue"]["pct_change"] == 0.2
    assert lift["orders"]["abs_change"] == 2.0
    assert lift["orders"]["pct_change"] == 0.25


def test_compute_lift_for_window_deterministic() -> None:
    lift = compute_lift_for_window(
        _df(),
        promo_start="2025-01-08T00:00:00Z",
        promo_end="2025-01-09T23:59:59Z",
        baseline_method="custom",
        baseline_start="2025-01-01T00:00:00Z",
        baseline_end="2025-01-02T23:59:59Z",
    )
    assert lift["revenue"]["promo"] == 57.0
    assert lift["revenue"]["baseline"] == 29.0
    assert lift["revenue"]["abs_change"] == 28.0
    assert round((lift["revenue"]["pct_change"] or 0.0) * 100, 2) == 96.55
