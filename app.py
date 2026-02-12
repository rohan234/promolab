"""Streamlit entrypoint for PromoLab."""

from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from promolab.io import DataValidationError, load_transactions
from promolab.llm import generate_explanation
from promolab.metrics import compute_lift_for_window, daily_revenue_series, get_baseline_window
from promolab.report import generate_markdown_report

st.set_page_config(page_title="PromoLab", layout="wide")

REQUIRED_COLUMNS = [
    "timestamp",
    "order_id",
    "item_name",
    "quantity",
    "unit_price",
    "discount_amount",
    "refund_amount",
]
OPTIONAL_COLUMNS = ["cogs_amount"]


@st.cache_data
def _template_csv_bytes() -> bytes:
    template = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-05T12:30:00Z",
                "order_id": "A1001",
                "item_name": "Iced Latte",
                "quantity": 1,
                "unit_price": 5.5,
                "discount_amount": 0.5,
                "refund_amount": 0.0,
                "cogs_amount": 2.0,
            },
            {
                "timestamp": "2026-01-05T12:31:00Z",
                "order_id": "A1001",
                "item_name": "Blueberry Muffin",
                "quantity": 1,
                "unit_price": 3.25,
                "discount_amount": 0.0,
                "refund_amount": 0.0,
                "cogs_amount": 1.1,
            },
        ]
    )
    buffer = io.StringIO()
    template.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _example_dataset_bytes() -> tuple[bytes, str]:
    sample_path = Path("sample_data/transactions.csv")
    if sample_path.exists():
        return sample_path.read_bytes(), sample_path.name
    return _template_csv_bytes(), "example_transactions.csv"


def _suggested_window(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = max_ts.floor("D")
    start = max(min_ts.floor("D"), end - pd.Timedelta(days=2))
    return start, end
st.title("PromoLab")
st.caption("Upload → select promo window → baseline + lift, charts, diagnostics, AI explanation, and pull-forward checks")


def _to_utc_day_bounds(start_date, end_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(start_date, tz="UTC")
    e = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return s, e


def _window_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.floor("D") - start.floor("D")).days + 1)


def _gap_days(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    all_days = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    if df.empty:
        return len(all_days)
    existing = set(pd.to_datetime(df["timestamp"], utc=True).dt.floor("D").unique())
    return int(sum(day not in existing for day in all_days))


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df.loc[(ts >= start) & (ts <= end)].copy()


def _line_revenue(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(((df["quantity"] * df["unit_price"]) - df["discount_amount"] - df["refund_amount"]).sum())


def _top_item_drivers(promo_df: pd.DataFrame, baseline_df: pd.DataFrame, promo_days: int, baseline_days: int) -> pd.DataFrame:
    if promo_df.empty and baseline_df.empty:
        return pd.DataFrame(columns=["item_name", "delta_revenue"])

    promo_work = promo_df.copy()
    base_work = baseline_df.copy()

    for frame in (promo_work, base_work):
        frame["line_revenue"] = (frame["quantity"] * frame["unit_price"]) - frame["discount_amount"] - frame["refund_amount"]

    promo = promo_work.groupby("item_name", as_index=False)["line_revenue"].sum().rename(columns={"line_revenue": "promo_revenue"})
    base = base_work.groupby("item_name", as_index=False)["line_revenue"].sum().rename(columns={"line_revenue": "baseline_total"})

    merged = promo.merge(base, on="item_name", how="outer").fillna(0.0)
    merged["baseline_for_promo_days"] = merged["baseline_total"] / max(baseline_days, 1) * max(promo_days, 1)
    merged["delta_revenue"] = merged["promo_revenue"] - merged["baseline_for_promo_days"]
    return merged.sort_values("delta_revenue", ascending=False)


def _pull_forward_check(
    df: pd.DataFrame,
    promo_start: pd.Timestamp,
    promo_end: pd.Timestamp,
    baseline_df: pd.DataFrame,
    baseline_days: int,
) -> tuple[dict[str, float | bool], pd.DataFrame]:
    post_start = promo_end.floor("D") + pd.Timedelta(days=1)
    post_end = post_start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)

    promo_df = _slice(df, promo_start, promo_end)
    post_df = _slice(df, post_start, post_end)

    baseline_daily_revenue = _line_revenue(baseline_df) / max(baseline_days, 1)
    promo_days = _window_days(promo_start, promo_end)
    post_days = _window_days(post_start, post_end)

    promo_actual = _line_revenue(promo_df)
    post_actual = _line_revenue(post_df)
    promo_expected = baseline_daily_revenue * promo_days
    post_expected = baseline_daily_revenue * post_days

    promo_uplift = promo_actual - promo_expected
    post_gap = post_actual - post_expected

    risk_flag = promo_uplift > 0 and post_gap < 0 and (post_expected == 0 or (post_gap / post_expected) <= -0.1)

    summary = {
        "promo_actual": promo_actual,
        "promo_expected": promo_expected,
        "post_actual": post_actual,
        "post_expected": post_expected,
        "promo_uplift": promo_uplift,
        "post_gap": post_gap,
        "risk_flag": risk_flag,
    }

    chart_df = pd.DataFrame(
        [
            {"period": "Promo window", "series": "Actual", "revenue": promo_actual},
            {"period": "Promo window", "series": "Baseline expected", "revenue": promo_expected},
            {"period": "Post 7 days", "series": "Actual", "revenue": post_actual},
            {"period": "Post 7 days", "series": "Baseline expected", "revenue": post_expected},
        ]
    )

    return summary, chart_df


st.title("PromoLab")
st.caption("Deterministic promo lift analysis for small businesses—no made-up numbers.")

with st.container(border=True):
    st.subheader("Getting started (~2 minutes)")
    step_cols = st.columns(3)
    step_cols[0].markdown("**1) Upload CSV**  \\nUse your POS export or our template.")
    step_cols[1].markdown("**2) Pick promo dates**  \\nChoose when your promotion ran.")
    step_cols[2].markdown("**3) Review + export**  \\nSee lift, drivers, diagnostics, and download report.")
    st.info("Outputs include KPI lift, top item drivers, validity diagnostics, pull-forward check, and exportable report.")

utility_cols = st.columns(1)
example_bytes, example_name = _example_dataset_bytes()

utility_cols = st.columns(2)
example_bytes, example_name = _example_dataset_bytes()
utility_cols[0].download_button(
    "Download template CSV",
    data=_template_csv_bytes(),
    file_name="promolab_template.csv",
    mime="text/csv",
)
utility_cols[1].download_button(
    "Download example dataset",
    data=example_bytes,
    file_name=example_name,
    mime="text/csv",
)

st.subheader("Step 1: Upload your transactions file")
st.markdown(
    "Upload a CSV export from your POS system with one row per item sold. "
    "Required columns are: `timestamp`, `order_id`, `item_name`, `quantity`, `unit_price`, "
    "`discount_amount`, and `refund_amount`."
)

uploaded = st.file_uploader(
    "Choose your transactions CSV file",
    type=["csv"],
    help="CSV only. For best performance, keep file size under ~50MB.",
)

with st.expander("What should my CSV look like?"):
    st.code(
        "timestamp,order_id,item_name,quantity,unit_price,discount_amount,refund_amount,cogs_amount\n"
        "2026-01-05T12:30:00Z,A1001,Iced Latte,1,5.50,0.50,0.00,2.00",
        language="csv",
    )
    st.download_button(
        "Download template CSV",
        data=_template_csv_bytes(),
        file_name="promolab_template.csv",
        mime="text/csv",
        key="template_in_schema_expander",
    )
with st.expander("What should my CSV look like?"):
    st.markdown("**Required columns**")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.markdown("**Optional columns**")
    st.code(", ".join(OPTIONAL_COLUMNS), language="text")
    st.markdown("**Example rows**")
    st.code(_template_csv_bytes().decode("utf-8"), language="csv")

with st.expander("Common export tips"):
    st.markdown("- Ensure `timestamp` is ISO or a recognizable datetime.")
    st.markdown("- Discounts/refunds should be numeric (`0` is allowed).")
    st.markdown("- Use one row per line item; the same `order_id` can repeat.")


if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = load_transactions(uploaded)
except DataValidationError as exc:
    msg = str(exc)
    if "missing column:" in msg:
        st.error(f"Your CSV is missing required columns: {msg.split('missing column:', 1)[1].strip()}")
    else:
        st.error(f"We couldn't read that file: {msg}")
    st.download_button(
        "Download template CSV",
        data=_template_csv_bytes(),
        file_name="promolab_template.csv",
        mime="text/csv",
        key="error_template_download",
    )
    st.error(str(exc))
    st.stop()

min_ts = pd.to_datetime(df["timestamp"], utc=True).min().floor("D")
max_ts = pd.to_datetime(df["timestamp"], utc=True).max().floor("D")

a, b, c = st.columns(3)
a.metric("Date range", f"{min_ts.date()} → {max_ts.date()}")
b.metric("Unique orders", int(df["order_id"].nunique()))
c.metric("Rows", int(len(df)))

missing_key_values = int(df[REQUIRED_COLUMNS].isna().sum().sum())
st.info(f"Data health check: missing values across required columns = **{missing_key_values}**")
with st.expander("Preview first 5 rows"):
    st.dataframe(df.head(5), use_container_width=True)

suggest_window = st.toggle("Show me a suggested promo window", value=False, key = "AI Suggest Window")
if suggest_window:
    suggested_start, suggested_end = _suggested_window(min_ts, max_ts)
    default_start, default_end = suggested_start.date(), suggested_end.date()
    st.info(f"Suggested window: {default_start} to {default_end} (recent 3-day window).")
else:
    default_end = max_ts.date()
    default_start = max(min_ts, max_ts - pd.Timedelta(days=6)).date()

default_end = max_ts.date()
default_start = max(min_ts, max_ts - pd.Timedelta(days=6)).date()

promo_start_date, promo_end_date = st.date_input(
    "Promo window (start and end)",
    value=(default_start, default_end),
    min_value=min_ts.date(),
    max_value=max_ts.date(),
)

if promo_start_date > promo_end_date:
    st.warning("Promo window invalid: start date must be on or before end date.")
    st.error("Promo start must be on or before promo end.")
    st.stop()

promo_start, promo_end = _to_utc_day_bounds(promo_start_date, promo_end_date)
promo_days = _window_days(promo_start, promo_end)
if promo_days < 2:
    st.warning("A good promo window is usually at least 2–3 days so lift is less noisy.")

baseline_method = st.selectbox(
    "Baseline method",
    options=["matched_4w", "last_28d", "custom"],
    format_func=lambda x: {
        "matched_4w": "Matched weekdays (previous 4 weeks)",
        "last_28d": "Last 28 days",
        "custom": "Custom baseline",
    }[x],
)

has_cogs = "cogs_amount" in df.columns and df["cogs_amount"].fillna(0).sum() > 0
margin_assumption = 0.6
if not has_cogs:
    margin_assumption = st.slider("Margin assumption (used when COGS missing)", 0.0, 1.0, 0.6, 0.05)

baseline_start = baseline_end = None
baseline_meta: dict[str, object] | None = None
if baseline_method == "custom":
    default_base_start = max(min_ts, promo_start - pd.Timedelta(days=28)).date()
    default_base_end = min(max_ts, promo_start - pd.Timedelta(days=1)).date()
    base_start_date, base_end_date = st.date_input(
        "Custom baseline window",
        value=(default_base_start, default_base_end),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )
    if base_start_date > base_end_date:
        st.warning("Baseline window invalid: start date must be on or before end date.")
        st.error("Baseline start must be on or before baseline end.")
        st.stop()
    baseline_start, baseline_end = _to_utc_day_bounds(base_start_date, base_end_date)

lift = compute_lift_for_window(
    df,
    promo_start,
    promo_end,
    baseline_method=baseline_method,
    margin_assumption=margin_assumption,
    baseline_start=baseline_start,
    baseline_end=baseline_end,
)

if baseline_method == "custom":
    baseline_days = _window_days(baseline_start, baseline_end)
    baseline_df = _slice(df, baseline_start, baseline_end)
else:
    baseline_meta = get_baseline_window(df, promo_start, promo_end, method=baseline_method)
    if baseline_meta["type"] == "range":
        b_start, b_end = baseline_meta["start"], baseline_meta["end"]
        baseline_df = _slice(df, b_start, b_end)
        baseline_days = _window_days(b_start, b_end)
    else:
        windows = baseline_meta["windows"]
        baseline_df = pd.concat([_slice(df, w.start, w.end) for w in windows], ignore_index=True)
        baseline_days = sum(_window_days(w.start, w.end) for w in windows)

promo_df = _slice(df, promo_start, promo_end)
item_delta = _top_item_drivers(promo_df, baseline_df, promo_days=promo_days, baseline_days=max(baseline_days, 1))
cannibalization_summary, cannibalization_chart = _pull_forward_check(df, promo_start, promo_end, baseline_df, baseline_days)

warnings: list[str] = []
if baseline_days < 14 or baseline_df.empty:
    warnings.append("Baseline coverage too small for reliable comparison.")
if promo_days < 3:
    warnings.append("Promo window is very short; lift may be noisy.")
if _gap_days(df, promo_start, promo_end) > 0:
    warnings.append("Promo window contains missing data days (gaps).")
if bool(cannibalization_summary["risk_flag"]):
    warnings.append(
        "Potential pull-forward risk: promo outperformed baseline, but the 7 days after promo underperformed baseline expectation."
    )

promo_daily = daily_revenue_series(df, promo_start, promo_end).assign(period="promo")

st.subheader("KPI table")
kpi_rows = []
for metric in ["revenue", "orders", "transactions", "aov", "discount_rate", "refund_rate", "gross_profit"]:
    m = lift[metric]
    pct = "n/a" if m["pct_change"] is None else f"{m['pct_change'] * 100:.2f}%"
    kpi_rows.append(
        {
            "metric": metric,
            "promo": m["promo"],
            "baseline": m["baseline"],
            "lift_abs": m["abs_change"],
            "lift_pct": pct,
        }
    )
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)

st.subheader("Revenue by day")
promo_daily = daily_revenue_series(df, promo_start, promo_end).assign(period="promo")

if baseline_method == "custom" or (baseline_meta and baseline_meta["type"] == "range"):
    if baseline_method == "custom":
        b_start, b_end = baseline_start, baseline_end
    else:
        b_start, b_end = baseline_meta["start"], baseline_meta["end"]
    base_daily = daily_revenue_series(df, b_start, b_end).assign(period="baseline")
else:
    parts = [daily_revenue_series(df, w.start, w.end) for w in baseline_meta["windows"]]
    base_daily = pd.concat(parts, ignore_index=True).groupby("date", as_index=False)["revenue"].mean().assign(period="baseline")

kpi_rows = []
for metric in ["revenue", "orders", "transactions", "aov", "discount_rate", "refund_rate", "gross_profit"]:
    m = lift[metric]
    pct = "n/a" if m["pct_change"] is None else f"{m['pct_change'] * 100:.2f}%"
    kpi_rows.append({"metric": metric, "promo": m["promo"], "baseline": m["baseline"], "lift_abs": m["abs_change"], "lift_pct": pct})
kpi_df = pd.DataFrame(kpi_rows)

results_tab, charts_tab, diagnostics_tab, export_tab = st.tabs(["Results", "Charts", "Diagnostics", "Export"])

with results_tab:
    st.subheader("KPI summary + lift")
    st.dataframe(kpi_df, use_container_width=True)
    st.subheader("Top item drivers (Δ revenue)")
    st.dataframe(item_delta[["item_name", "delta_revenue"]].head(15), use_container_width=True)

with charts_tab:
    st.subheader("Daily revenue")
    daily_plot = pd.concat([promo_daily, base_daily], ignore_index=True)
    if daily_plot.empty:
        st.info("No daily revenue data available for selected windows.")
    else:
        fig = px.line(daily_plot, x="date", y="revenue", color="period", title="Daily revenue: promo vs baseline")
        fig.add_vrect(x0=promo_start.floor("D"), x1=promo_end.floor("D"), fillcolor="green", opacity=0.08, line_width=0)
        st.plotly_chart(fig, use_container_width=True, key="daily_revenue_chart")

    st.subheader("Top item lift drivers")
    if item_delta.empty:
        st.info("No item-level data available.")
    else:
        fig_bar = px.bar(item_delta.head(15), x="item_name", y="delta_revenue", title="Top items driving revenue lift")
        st.plotly_chart(fig_bar, use_container_width=True, key="top_item_drivers_chart")

with diagnostics_tab:
    st.info(
        "Pull-forward risk means sales may have shifted earlier due to the promo; "
        "we check for a post-promo dip vs baseline."
    )
    st.markdown(f"**Pull-forward flag:** {'YES' if cannibalization_summary['risk_flag'] else 'NO'}")
    fig_cannibalization = px.bar(
        cannibalization_chart,
        x="period",
        y="revenue",
        color="series",
        barmode="group",
        title="Promo window and post 7-day revenue vs baseline expectation",
    )
    st.plotly_chart(fig_cannibalization, use_container_width=True, key="cannibalization_chart")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No major validity warnings detected.")

    st.subheader("AI explanation")
    context_text = st.text_area("Optional context", placeholder="Promo was 10% off drinks Fri–Sun")
    ai_text = ""
    use_ai = st.toggle("Generate AI explanation", value=False, key="toggle_ai_explanation_diag")

    if use_ai:
        summary_payload = {
            "promo_window": {"start": str(promo_start), "end": str(promo_end)},
            "baseline_method": baseline_method,
            "margin_assumption": margin_assumption,
            "kpi_lift": lift,
            "top_drivers": item_delta.head(5).to_dict(orient="records"),
            "warnings": warnings,
            "cannibalization_check": cannibalization_summary,
            "context": context_text,
        }
        with st.spinner("Generating explanation..."):
            ai_text = generate_explanation(summary_payload)
        st.markdown(ai_text)
    st.session_state["promolab_ai_text"] = ai_text

with export_tab:
    ai_text = st.session_state.get("promolab_ai_text") or None
    report_md = generate_markdown_report(
        promo_start=promo_start,
        promo_end=promo_end,
        baseline_method=baseline_method,
        lift=lift,
        drivers=item_delta,
        warnings=warnings,
        ai_explanation=ai_text,
        cannibalization_summary=cannibalization_summary,
    )
    st.download_button(
        "Download Promo Report (.md)",
        data=report_md,
        file_name="promolab_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download KPI results (.csv)",
        data=kpi_df.to_csv(index=False),
        file_name="promolab_kpis.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "How this differs from ChatGPT: PromoLab computes deterministic KPIs directly from your uploaded transactions "
    "and applies guardrail diagnostics (baseline quality, gaps, pull-forward) before interpretation."
)

daily_plot = pd.concat([promo_daily, base_daily], ignore_index=True)
if daily_plot.empty:
    st.info("No daily revenue data available for selected windows.")
else:
    fig = px.line(daily_plot, x="date", y="revenue", color="period", title="Daily revenue: promo vs baseline")
    fig.add_vrect(x0=promo_start.floor("D"), x1=promo_end.floor("D"), fillcolor="green", opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top item drivers (Δ revenue)")
item_delta = _top_item_drivers(promo_df, baseline_df, promo_days=promo_days, baseline_days=max(baseline_days, 1))
if item_delta.empty:
    st.info("No item-level data available.")
else:
    fig_bar = px.bar(item_delta.head(15), x="item_name", y="delta_revenue", title="Top items driving revenue lift")
    st.plotly_chart(fig_bar, use_container_width=True)

cannibalization_summary, cannibalization_chart = _pull_forward_check(df, promo_start, promo_end, baseline_df, baseline_days)
st.subheader("Pull-forward / cannibalization check")
fig_cannibalization = px.bar(
    cannibalization_chart,
    x="period",
    y="revenue",
    color="series",
    barmode="group",
    title="Promo window and post 7-day revenue vs baseline expectation",
)
st.plotly_chart(fig_cannibalization, use_container_width=True)

warnings: list[str] = []
if baseline_days < 14 or baseline_df.empty:
    warnings.append("Baseline coverage too small for reliable comparison.")
if promo_days < 3:
    warnings.append("Promo window is very short; lift may be noisy.")
if _gap_days(df, promo_start, promo_end) > 0:
    warnings.append("Promo window contains missing data days (gaps).")
if bool(cannibalization_summary["risk_flag"]):
    warnings.append(
        "Potential pull-forward risk: promo outperformed baseline, but the 7 days after promo underperformed baseline expectation."
    )

if warnings:
    for w in warnings:
        st.warning(w)
else:
    st.success("No major validity warnings detected.")

st.subheader("AI explanation")
use_ai = st.toggle("Generate AI explanation", value=False, key="toggle_ai_explanation_main")
context_text = st.text_area("Optional context", placeholder="Promo was 10% off drinks Fri–Sun", key="context_text")
ai_text = ""
if use_ai:
    summary_payload = {
        "promo_window": {"start": str(promo_start), "end": str(promo_end)},
        "baseline_method": baseline_method,
        "margin_assumption": margin_assumption,
        "kpi_lift": lift,
        "top_drivers": item_delta.head(5).to_dict(orient="records"),
        "warnings": warnings,
        "cannibalization_check": cannibalization_summary,
        "context": context_text,
    }
    with st.spinner("Generating explanation..."):
        ai_text = generate_explanation(summary_payload)
    st.markdown(ai_text)

st.subheader("Export report")
report_md = generate_markdown_report(
    promo_start=promo_start,
    promo_end=promo_end,
    baseline_method=baseline_method,
    lift=lift,
    drivers=item_delta,
    warnings=warnings,
    ai_explanation=ai_text if use_ai else None,
    cannibalization_summary=cannibalization_summary,
)
st.download_button(
    "Download Promo Report (.md)",
    data=report_md,
    file_name="promolab_report.md",
    mime="text/markdown",
)
