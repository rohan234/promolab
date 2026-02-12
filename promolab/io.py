"""CSV ingestion and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd

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
NUMERIC_COLUMNS = ["quantity", "unit_price", "discount_amount", "refund_amount", "cogs_amount"]


@dataclass
class DataValidationError(ValueError):
    """Raised when incoming CSV data fails schema or type validation."""


def load_transactions(file: str | BinaryIO) -> pd.DataFrame:
    """Load and validate transactions CSV.

    Parameters
    ----------
    file:
        Path or uploaded file object accepted by pandas.
    """
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("uploaded file is empty") from exc
    except FileNotFoundError as exc:
        raise DataValidationError("file not found") from exc
    except Exception as exc:  # pragma: no cover
        raise DataValidationError(f"unable to read csv: {exc}") from exc

    if df.empty:
        raise DataValidationError("uploaded file has no rows")

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"missing column: {', '.join(missing)}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if parsed_ts.isna().any():
        raise DataValidationError("invalid timestamp values in column 'timestamp'")
    df["timestamp"] = parsed_ts

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            raise DataValidationError(f"invalid numeric values in column '{col}'")
        df[col] = coerced.astype(float)

    if (df["quantity"] <= 0).any():
        raise DataValidationError("quantity must be positive")

    for col in ["unit_price", "discount_amount", "refund_amount", "cogs_amount"]:
        if (df[col] < 0).any():
            raise DataValidationError(f"column '{col}' must be nonnegative")

    return df[
        [
            "timestamp",
            "order_id",
            "item_name",
            "quantity",
            "unit_price",
            "discount_amount",
            "refund_amount",
            "cogs_amount",
        ]
    ].copy()
