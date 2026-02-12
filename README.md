# PromoLab

PromoLab is a Streamlit app for deterministic promotion-impact analysis.

## Features (Milestones 0–6)
The Codex workspace blocks external package installs; run locally or via Streamlit Cloud to launch the UI.

- Upload and validate transactions CSV against a strict schema.
- Select a promo date window and baseline method (`matched_4w`, `last_28d`, `custom`).
- Compute KPI comparisons and lift (absolute + percent).
- Visualize daily revenue trends and top item drivers.
- Show guardrail warnings for low baseline coverage, short promo windows, and data gaps.
- Optional AI explanation layer (uses only provided computed numbers) with experiments + promo copy.
- Export a recruiter-friendly Markdown report.
- Pull-forward / cannibalization heuristic that checks post-promo 7-day underperformance vs baseline expectation.

## Transactions schema

Required columns:

- `timestamp` (ISO datetime)
- `order_id`
- `item_name`
- `quantity`
- `unit_price`
- `discount_amount`
- `refund_amount`

Optional columns:

- `cogs_amount` (if missing, app uses margin assumption)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## AI configuration (optional)

Set an API key to enable real LLM output in the app:

```bash
export PROMOLAB_LLM_API_KEY=your_key_here
# optional
export PROMOLAB_LLM_MODEL=gpt-4o-mini
```

If not set, the app returns a safe placeholder explanation.


Note: The Codex workspace may block external package installs; run locally or via Streamlit Cloud to launch the UI.

## Run tests

```bash
pytest
```

## Sample data

- `sample_data/transactions.csv`
- `sample_data/promos.csv`
