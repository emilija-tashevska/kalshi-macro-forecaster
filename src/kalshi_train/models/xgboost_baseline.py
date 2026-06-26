"""Classical ML baselines — XGBoost with temporal cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from kalshi_train.eval.metrics import MetricReport, compute_metrics
from kalshi_train.eval.splits import expanding_window_cv, split_xy


@dataclass(frozen=True, slots=True)
class XGBoostCVResult:
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    feature_importance: pd.Series
    fold_metrics: list[MetricReport]
    mean_metrics: MetricReport


def _default_xgb_params() -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def _resolved_params(y: np.ndarray, xgb_params: dict[str, Any] | None) -> dict[str, Any]:
    # NOTE: we deliberately do NOT use scale_pos_weight here. Class-weight
    # reweighting optimizes balanced error but destroys probability
    # calibration, which is exactly what our proper scoring rules (Brier,
    # log loss) reward. We keep natural class frequencies and rely on
    # post-hoc calibration instead.
    _ = y
    params = _default_xgb_params()
    if xgb_params:
        params.update(xgb_params)
    return params


def train_xgboost_temporal_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    n_splits: int = 5,
    xgb_params: dict[str, Any] | None = None,
) -> XGBoostCVResult:
    """Expanding-window CV with out-of-fold predictions for diagnostics."""
    ordered = df.sort_values("as_of_date")
    x, y = split_xy(ordered, feature_cols, label_col=label_col)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(x)

    tscv = expanding_window_cv(len(ordered), n_splits=n_splits)
    oof = np.full(len(ordered), np.nan)
    fold_metrics: list[MetricReport] = []
    importances = np.zeros(len(feature_cols), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(x_imputed)):
        params = _resolved_params(y[train_idx], xgb_params)
        model = xgb.XGBClassifier(**params)
        model.fit(x_imputed[train_idx], y[train_idx])
        probs = model.predict_proba(x_imputed[test_idx])[:, 1]
        oof[test_idx] = probs
        fold_metrics.append(compute_metrics(y[test_idx], probs))
        importances += model.feature_importances_
        _ = fold_idx

    valid = ~np.isnan(oof)
    mean_metrics = compute_metrics(y[valid], oof[valid])
    importance = pd.Series(importances / max(len(fold_metrics), 1), index=feature_cols)
    importance = importance.sort_values(ascending=False)

    return XGBoostCVResult(
        oof_predictions=oof,
        oof_indices=np.arange(len(ordered)),
        feature_importance=importance,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
    )


def train_xgboost_final(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    xgb_params: dict[str, Any] | None = None,
) -> tuple[xgb.XGBClassifier, SimpleImputer]:
    """Fit on the full training split for held-out test evaluation."""
    x, y = split_xy(train_df, feature_cols, label_col=label_col)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(x)
    params = _resolved_params(y, xgb_params)
    model = xgb.XGBClassifier(**params)
    model.fit(x_imputed, y)
    return model, imputer


def predict_proba(
    model: xgb.XGBClassifier,
    imputer: SimpleImputer,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    x = imputer.transform(df[feature_cols].to_numpy(dtype=float))
    probs: np.ndarray = model.predict_proba(x)[:, 1]
    return probs


# ── Probability calibration (Platt scaling) ────────────────────────────


def _to_logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.asarray(np.log(p / (1.0 - p)), dtype=float)
    return logit.reshape(-1, 1)


def fit_platt_calibrator(
    probs: np.ndarray, labels: np.ndarray
) -> LogisticRegression | None:
    """Fit Platt scaling (1-D logistic on the logit of the model output).

    Calibrate on a held-out split (e.g. validation) whose period matches
    the test period, to correct over/under-confidence. Returns ``None``
    when the calibration set has a single class (can't fit), so the caller
    falls back to raw probabilities.
    """
    if len(np.unique(labels)) < 2:
        return None
    lr = LogisticRegression()
    lr.fit(_to_logit(probs), labels)
    return lr


def apply_calibrator(
    calibrator: LogisticRegression | None, probs: np.ndarray
) -> np.ndarray:
    """Apply a fitted Platt calibrator; identity when ``calibrator`` is None."""
    if calibrator is None:
        return probs
    calibrated: np.ndarray = calibrator.predict_proba(_to_logit(probs))[:, 1]
    return calibrated
