"""Ham du doan backorder dung model A - Random Forest va preprocessing tu notebook."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def load_assets() -> dict[str, Any]:
    """Tai mo hinh va cac buoc tien xu ly da duoc fit tu tap train."""
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "metadata": metadata,
        "imputer": joblib.load(MODEL_DIR / "imputer_median.joblib"),
        "scaler": joblib.load(MODEL_DIR / "scaler_luong_A_robust.joblib"),
        "model": joblib.load(MODEL_DIR / "model_A_RandomForest.joblib"),
    }


def clean_input_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Kiem tra cot, ep kieu so, xu ly -99 va gia tri vo cuc."""
    feature_cols = metadata["feature_cols"]
    missing_cols = [c for c in feature_cols if c not in frame.columns]
    if missing_cols:
        raise ValueError(
            "File du lieu thieu cac cot bat buoc: " + ", ".join(missing_cols)
        )

    X = frame[feature_cols].copy()
    warnings_list: list[str] = []

    # Các cột cờ trong bộ dữ liệu gốc có thể được lưu dưới dạng Yes/No,
    # trong khi bộ clu.zip đã mã hóa chúng thành 1/0. Chuẩn hóa cả hai dạng.
    binary_cols = {
        "potential_issue", "deck_risk", "oe_constraint",
        "ppap_risk", "stop_auto_buy", "rev_stop",
    }
    binary_map = {
        "yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
        "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0,
    }

    for col in feature_cols:
        before_missing = int(X[col].isna().sum())
        if col in binary_cols:
            numeric_values = pd.to_numeric(X[col], errors="coerce")
            text_values = X[col].astype("string").str.strip().str.casefold()
            mapped_values = text_values.map(binary_map)
            X[col] = numeric_values.fillna(mapped_values)
            unknown_mask = X[col].isna() & frame[col].notna()
            unknown_count = int(unknown_mask.sum())
            if unknown_count:
                examples = frame.loc[unknown_mask, col].astype(str).drop_duplicates().head(3).tolist()
                warnings_list.append(
                    f"Cot {col}: co {unknown_count} gia tri co dang khong nhan dien duoc "
                    f"({', '.join(examples)}); da chuyen thanh thieu."
                )
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            after_missing = int(X[col].isna().sum())
            if after_missing > before_missing:
                warnings_list.append(
                    f"Cot {col}: co gia tri khong phai so, da chuyen thanh thieu."
                )

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in metadata.get("sentinel_cols", []):
        if col in X.columns:
            sentinel_count = int((X[col] == -99).sum())
            if sentinel_count:
                warnings_list.append(f"Cot {col}: da xu ly {sentinel_count} gia tri -99 nhu du lieu thieu.")
                X[col] = X[col].replace(-99, np.nan)

    return X, warnings_list


def create_traditional_features(X_clean: pd.DataFrame) -> pd.DataFrame:
    """Phai dong nhat voi ham Feature Engineering trong notebook."""
    X = X_clean.copy()
    X["net_inventory_in_transit"] = X["national_inv"] + X["in_transit_qty"]
    X["forecast_inventory_gap_3m"] = X["forecast_3_month"] - X["national_inv"]
    X["sales_forecast_gap_3m"] = X["sales_3_month"] - X["forecast_3_month"]
    X["inventory_to_forecast_3m"] = X["national_inv"] / (X["forecast_3_month"].abs() + 1)
    X["past_due_to_transit_ratio"] = X["pieces_past_due"] / (X["in_transit_qty"] + 1)
    X["supplier_perf_mean"] = X[["perf_6_month_avg", "perf_12_month_avg"]].mean(axis=1)
    return X


def predict_frame(
    frame: pd.DataFrame,
    assets: dict[str, Any],
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Tra ve bang ket qua du doan cho mot hoac nhieu san pham."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Nguong du doan phai nam trong khoang 0 den 1.")

    metadata = assets["metadata"]
    X_raw, warnings_list = clean_input_frame(frame, metadata)
    X_imputed = pd.DataFrame(
        assets["imputer"].transform(X_raw),
        columns=metadata["feature_cols"],
        index=frame.index,
    )
    X_engineered = create_traditional_features(X_imputed)
    X_scaled = assets["scaler"].transform(X_engineered)
    probabilities = assets["model"].predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= float(threshold)).astype(int)

    id_col = metadata.get("id_col", "sku")
    if id_col in frame.columns:
        identifiers = frame[id_col].astype(str).values
    else:
        identifiers = [f"San_pham_{i + 1}" for i in range(len(frame))]

    result = pd.DataFrame({
        "sku": identifiers,
        # Giữ xác suất đầy đủ, không làm tròn trước khi xuất CSV.
        # Nhãn du_doan được tính từ xác suất gốc và ngưỡng chính xác.
        "xac_suat_thieu_hang": probabilities,
        "nguong_ap_dung": np.full(len(frame), float(threshold), dtype=float),
        "du_doan": predictions,
        "ket_luan": np.where(predictions == 1, "Can ra soat nguy co thieu hang", "Chua vuot nguong canh bao"),
    })
    result["goi_y_xu_ly"] = [
        build_recommendation(row, float(prob), int(pred), float(threshold))
        for (_, row), prob, pred in zip(frame.iterrows(), probabilities, predictions)
    ]
    return result, warnings_list


def build_recommendation(row: pd.Series, probability: float, prediction: int, threshold: float) -> str:
    """Goi y kiem tra theo quy tac don gian; khong tu dong tinh so luong dat hang."""
    if prediction == 0:
        return "Theo doi dinh ky; xac suat hien tai chua vuot nguong canh bao."

    actions = ["Uu tien kiem tra san pham co nguy co thieu hang."]
    inv = pd.to_numeric(row.get("national_inv", np.nan), errors="coerce")
    forecast = pd.to_numeric(row.get("forecast_3_month", np.nan), errors="coerce")
    in_transit = pd.to_numeric(row.get("in_transit_qty", np.nan), errors="coerce")
    overdue = pd.to_numeric(row.get("pieces_past_due", np.nan), errors="coerce")

    if pd.notna(inv) and pd.notna(forecast) and inv < forecast:
        actions.append("Ton kho hien tai thap hon du bao 3 thang; can ra soat ke hoach bo sung.")
    if pd.notna(in_transit) and in_transit > 0:
        actions.append("Kiem tra tien do giao cua luong hang dang van chuyen.")
    if pd.notna(overdue) and overdue > 0:
        actions.append("Theo doi hang qua han va lien he nha cung cap.")
    return " ".join(actions)
