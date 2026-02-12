from __future__ import annotations

import pandas as pd
import pytest

from promolab.io import DataValidationError, load_transactions


def _write(tmp_path, frame: pd.DataFrame) -> str:
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_missing_column(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "order_id": ["o1"],
            "item_name": ["item"],
            "quantity": [1],
            "unit_price": [10],
            "discount_amount": [0],
        }
    )
    with pytest.raises(DataValidationError, match="missing column: refund_amount"):
        load_transactions(_write(tmp_path, frame))


def test_bad_timestamp(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["bad-ts"],
            "order_id": ["o1"],
            "item_name": ["item"],
            "quantity": [1],
            "unit_price": [10],
            "discount_amount": [0],
            "refund_amount": [0],
        }
    )
    with pytest.raises(DataValidationError, match="invalid timestamp"):
        load_transactions(_write(tmp_path, frame))


def test_non_numeric_quantity(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "order_id": ["o1"],
            "item_name": ["item"],
            "quantity": ["oops"],
            "unit_price": [10],
            "discount_amount": [0],
            "refund_amount": [0],
        }
    )
    with pytest.raises(DataValidationError, match="invalid numeric values in column 'quantity'"):
        load_transactions(_write(tmp_path, frame))


def test_negative_values(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "order_id": ["o1"],
            "item_name": ["item"],
            "quantity": [1],
            "unit_price": [-1],
            "discount_amount": [0],
            "refund_amount": [0],
        }
    )
    with pytest.raises(DataValidationError, match="unit_price"):
        load_transactions(_write(tmp_path, frame))


def test_empty_file(tmp_path) -> None:
    path = tmp_path / "transactions.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_transactions(str(path))


def test_valid_file(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "order_id": ["o1"],
            "item_name": ["item"],
            "quantity": [2],
            "unit_price": [10.0],
            "discount_amount": [1.0],
            "refund_amount": [0.0],
        }
    )
    df = load_transactions(_write(tmp_path, frame))
    assert list(df.columns) == [
        "timestamp",
        "order_id",
        "item_name",
        "quantity",
        "unit_price",
        "discount_amount",
        "refund_amount",
        "cogs_amount",
    ]
    assert df.shape[0] == 1
