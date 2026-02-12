"""Deterministic KPI, baseline, and lift calculations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tz is None else out.tz_convert("UTC")


def _normalize_window(start: str | pd.Timestamp, end: str | pd.Timestamp) -> Window:
    s, e = _to_utc(start), _to_utc(end)
    if e < s:
        raise ValueError("window end must be >= window start")
    return Window(start=s, end=e)


def _slice(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    w = _normalize_window(start, end)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df.loc[(ts >= w.start) & (ts <= w.end)].copy()


def compute_kpis(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    margin_assumption: float = 0.6,
) -> dict[str, float]:
    """Compute KPI metrics for the selected window."""
    win_df = _slice(df, start, end)
    gross_sales = float((win_df["quantity"] * win_df["unit_price"]).sum())
    total_discount = float(win_df["discount_amount"].sum())
    total_refund = float(win_df["refund_amount"].sum())

    revenue = gross_sales - total_discount - total_refund
    orders = float(win_df["order_id"].nunique())
    aov = 0.0 if orders == 0 else revenue / orders
    discount_rate = 0.0 if gross_sales == 0 else total_discount / gross_sales
    refund_rate = 0.0 if gross_sales == 0 else total_refund / gross_sales

    if "cogs_amount" in win_df.columns and win_df["cogs_amount"].notna().any():
        gross_profit = revenue - float(win_df["cogs_amount"].fillna(0.0).sum())
    else:
        gross_profit = revenue * margin_assumption

    return {
        "revenue": float(revenue),
        "orders": float(orders),
        "aov": float(aov),
        "discount_rate": float(discount_rate),
        "refund_rate": float(refund_rate),
        "gross_profit": float(gross_profit),
        "transactions": float(len(win_df)),
        "gross_sales": float(gross_sales),
    }


def get_baseline_window(
    df: pd.DataFrame,
    promo_start: str | pd.Timestamp,
    promo_end: str | pd.Timestamp,
    method: str = "matched_4w",
) -> dict[str, object]:
    """Return baseline range definition.

    Returns dict:
      - matched_4w: {'type': 'windows', 'windows': list[Window]}
      - last_28d: {'type': 'range', 'start': Timestamp, 'end': Timestamp}
    """
    del df  # reserved for future data-aware strategies
    promo = _normalize_window(promo_start, promo_end)

    if method == "matched_4w":
        windows: list[Window] = []
        for i in range(1, 5):
            shift = pd.Timedelta(days=7 * i)
            windows.append(Window(start=promo.start - shift, end=promo.end - shift))
        return {"type": "windows", "windows": windows}

    if method == "last_28d":
        end = promo.start - pd.Timedelta(seconds=1)
        start = promo.start - pd.Timedelta(days=28)
        return {"type": "range", "start": start, "end": end}

    raise ValueError("unsupported baseline method")


def _aggregate_windows(df: pd.DataFrame, windows: list[Window], margin_assumption: float) -> dict[str, float]:
    if not windows:
        return {k: 0.0 for k in ["revenue", "orders", "aov", "discount_rate", "refund_rate", "gross_profit", "transactions", "gross_sales"]}

    snapshots = [compute_kpis(df, w.start, w.end, margin_assumption=margin_assumption) for w in windows]
    metrics = snapshots[0].keys()
    return {metric: float(sum(s[metric] for s in snapshots) / len(snapshots)) for metric in metrics}


def compute_lift(promo_kpis: dict[str, float], baseline_kpis: dict[str, float]) -> dict[str, dict[str, float | None]]:
    """Compute absolute and percent lift for each KPI."""
    result: dict[str, dict[str, float | None]] = {}
    for metric, promo_value in promo_kpis.items():
        baseline_value = baseline_kpis[metric]
        abs_change = float(promo_value - baseline_value)
        pct_change = None if baseline_value == 0 else float(abs_change / baseline_value)
        result[metric] = {
            "promo": float(promo_value),
            "baseline": float(baseline_value),
            "abs_change": abs_change,
            "pct_change": pct_change,
        }
    return result


def compute_lift_for_window(
    df: pd.DataFrame,
    promo_start: str | pd.Timestamp,
    promo_end: str | pd.Timestamp,
    baseline_method: str = "matched_4w",
    margin_assumption: float = 0.6,
    baseline_start: str | pd.Timestamp | None = None,
    baseline_end: str | pd.Timestamp | None = None,
) -> dict[str, dict[str, float | None]]:
    """Convenience wrapper used by UI."""
    promo_kpis = compute_kpis(df, promo_start, promo_end, margin_assumption=margin_assumption)

    if baseline_method == "custom":
        if baseline_start is None or baseline_end is None:
            raise ValueError("custom baseline requires baseline_start and baseline_end")
        baseline_kpis = compute_kpis(df, baseline_start, baseline_end, margin_assumption=margin_assumption)
    else:
        baseline = get_baseline_window(df, promo_start, promo_end, method=baseline_method)
        if baseline["type"] == "windows":
            baseline_kpis = _aggregate_windows(df, baseline["windows"], margin_assumption)
        else:
            baseline_kpis = compute_kpis(
                df,
                baseline["start"],
                baseline["end"],
                margin_assumption=margin_assumption,
            )

    return compute_lift(promo_kpis, baseline_kpis)


def daily_revenue_series(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    win_df = _slice(df, start, end)
    if win_df.empty:
        return pd.DataFrame(columns=["date", "revenue"])
    ts = pd.to_datetime(win_df["timestamp"], utc=True).dt.floor("D")
    work = win_df.copy()
    work["date"] = ts
    work["rev"] = (work["quantity"] * work["unit_price"]) - work["discount_amount"] - work["refund_amount"]
    return work.groupby("date", as_index=False)["rev"].sum().rename(columns={"rev": "revenue"})
