# src/backtest/prob_gate.py
"""Probability gate utilities for ATR breakout strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import math
import json
import uuid

import numpy as np
import pandas as pd


MODEL_DIR = Path("storage") / "models"
MODEL_VERSION = 1


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return float(math.log(p / (1.0 - p)))


def _ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def _model_path(model_id: str) -> Path:
    safe = "".join(ch for ch in model_id if ch.isalnum() or ch in ("-", "_", "."))
    return _ensure_model_dir() / f"{safe}.json"


def generate_model_id(prefix: str = "prob_gate") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}_{token}"


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / max(n, 1), adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    return rsi_vals.fillna(method="bfill").fillna(50.0)


def _compute_features(df: pd.DataFrame, breakout_n: int, exit_n: int, atr_n: int) -> pd.DataFrame:
    data = pd.DataFrame(index=df.index)
    atr = wilder_atr(df["high"], df["low"], df["close"], n=max(1, atr_n))
    atr = atr.replace(0.0, np.nan)
    prev_atr = atr.shift(1)
    roll_high = df["close"].rolling(breakout_n).max().shift(1)
    roll_low = df["close"].rolling(exit_n).min().shift(1)
    data["dip_from_high_atr"] = (roll_high - df["close"]) / atr
    data["distance_from_low_atr"] = (df["close"] - roll_low) / atr
    data["atr_pct"] = atr / df["close"].replace(0.0, np.nan)
    data["atr_trend"] = atr / prev_atr.replace(0.0, np.nan) - 1.0
    data["rsi_14"] = rsi(df["close"], 14) / 100.0
    data = data.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return data


def build_feature_frame(df: pd.DataFrame, params: object) -> pd.DataFrame:
    breakout_n = int(getattr(params, "breakout_n", 20) or 20)
    exit_n = int(getattr(params, "exit_n", 10) or 10)
    atr_n = int(getattr(params, "atr_n", 14) or 14)
    breakout_n = max(2, breakout_n)
    exit_n = max(2, exit_n)
    atr_n = max(2, atr_n)
    return _compute_features(df, breakout_n, exit_n, atr_n)


def prepare_training_data(
    df: pd.DataFrame,
    trades: Sequence[dict],
    params: object,
) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty or not trades:
        return pd.DataFrame(), pd.Series(dtype=float)
    feats = build_feature_frame(df, params)
    rows: List[pd.Series] = []
    labels: List[float] = []
    index = feats.index
    for trade in trades:
        entry_time = trade.get("entry_time")
        if entry_time is None:
            continue
        try:
            ts = pd.to_datetime(entry_time)
        except Exception:
            continue
        loc = index.get_indexer([ts], method="pad")
        if len(loc) == 0 or loc[0] == -1:
            continue
        feat_row = feats.iloc[loc[0]]
        rows.append(feat_row)
        labels.append(1.0 if float(trade.get("return_pct", 0.0) or 0.0) > 0.0 else 0.0)
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = pd.DataFrame(rows)
    y = pd.Series(labels, dtype=float)
    return X, y


def _standardise(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    features = [c for c in X.columns]
    values = X[features].astype(float).values
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std == 0.0] = 1.0
    values = (values - mean) / std
    values = np.nan_to_num(values, nan=0.0)
    return values, mean, std, features


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    lr: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    intercept = 0.0
    for _ in range(max_iter):
        logits = intercept + X.dot(weights)
        preds = _sigmoid(logits)
        error = preds - y
        grad_w = (X.T.dot(error) / n_samples) + l2 * weights
        grad_b = float(error.mean())
        max_grad = max(abs(grad_b), float(np.max(np.abs(grad_w))))
        intercept -= lr * grad_b
        weights -= lr * grad_w
        if max_grad < tol:
            break
    return weights, float(intercept)


def _fit_platt(logits: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    X = logits.reshape(-1, 1)
    w, b = _fit_logistic(X, y, l2=1e-3)
    slope = float(w[0]) if len(w) else 1.0
    return slope, b


@dataclass
class CalibratedLogisticModel:
    coef: List[float]
    intercept: float
    calibration_a: float
    calibration_b: float
    feature_names: List[str]
    feature_means: List[float]
    feature_stds: List[float]
    base_rate: float

    def to_dict(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "coef": self.coef,
            "intercept": self.intercept,
            "calibration_a": self.calibration_a,
            "calibration_b": self.calibration_b,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "base_rate": self.base_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibratedLogisticModel":
        return cls(
            coef=[float(x) for x in data.get("coef", [])],
            intercept=float(data.get("intercept", 0.0)),
            calibration_a=float(data.get("calibration_a", 1.0)),
            calibration_b=float(data.get("calibration_b", 0.0)),
            feature_names=[str(x) for x in data.get("feature_names", [])],
            feature_means=[float(x) for x in data.get("feature_means", [])],
            feature_stds=[float(x) for x in data.get("feature_stds", [])],
            base_rate=float(data.get("base_rate", 0.5)),
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if X.empty:
            return pd.Series(dtype=float)
        values = X[self.feature_names].astype(float).values
        values = (values - np.array(self.feature_means)) / np.array(self.feature_stds)
        values = np.nan_to_num(values, nan=0.0)
        logits = self.intercept + values.dot(np.array(self.coef))
        calibrated = self.calibration_b + self.calibration_a * logits
        probs = _sigmoid(calibrated)
        return pd.Series(probs, index=X.index, dtype=float)


def fit_model(X: pd.DataFrame, y: pd.Series) -> Optional[CalibratedLogisticModel]:
    if X.empty or y.empty:
        return None
    if len(np.unique(y)) < 2:
        base = float(y.mean()) if len(y) else 0.5
        intercept = _safe_logit(base)
        feats, mean, std, names = _standardise(X)
        zeros = [0.0 for _ in names]
        return CalibratedLogisticModel(zeros, intercept, 1.0, 0.0, names, mean.tolist(), std.tolist(), base)
    feats, mean, std, names = _standardise(X)
    weights, intercept = _fit_logistic(feats, y.values.astype(float))
    logits = intercept + feats.dot(weights)
    a, b = _fit_platt(logits, y.values.astype(float))
    base = float(y.mean())
    return CalibratedLogisticModel(weights.tolist(), intercept, a, b, names, mean.tolist(), std.tolist(), base)


def save_model(model_id: str, model: CalibratedLogisticModel) -> str:
    path = _model_path(model_id)
    payload = model.to_dict()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def load_model(model_id: str) -> CalibratedLogisticModel:
    path = _model_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"prob_gate model '{model_id}' not found")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibratedLogisticModel.from_dict(data)


def score_probabilities(df: pd.DataFrame, params: object, model_id: str) -> pd.Series:
    model = load_model(model_id)
    feats = build_feature_frame(df, params)
    if feats.empty:
        return pd.Series(dtype=float)
    scores = model.predict_proba(feats)
    scores = scores.reindex(df.index).fillna(method="ffill")
    return scores.fillna(model.base_rate)

