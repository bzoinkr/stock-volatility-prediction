import os
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pipelines.prediction.ridgeRegression.eval_social_pipeline import (
    compute_stats,
    print_stats,
    run_eval_pipeline,
)
from pipelines.prediction.ridgeRegression.ridgeCreateModel_social_pipeline import (
    _validate_weights,
    build_dataset,
    build_prediction_rows,
    load_json,
    run_pipeline,
)

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("Install xgboost to use this pipeline.") from exc


def _save_model(model, model_dir: str, filename: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def _load_model(model_dir: str, filename: str):
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_xgboost_model(n_samples: int, seed: int = 42) -> XGBRegressor:
    if n_samples < 600:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.65,
            colsample_bytree=0.65,
            reg_alpha=0.6,
            reg_lambda=3.0,
            gamma=0.2,
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=260,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=2.0,
        gamma=0.1,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )


def _build_svr_model(n_samples: int) -> Pipeline:
    if n_samples < 600:
        c, eps = 0.8, 0.04
    else:
        c, eps = 1.5, 0.03
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=c, epsilon=eps, gamma="scale")),
        ]
    )


def train_model(model_type: str, X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    if model_type == "xgboost":
        # Keep the same bundle interface as Ridge: scaler + model.
        scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(X)
        model = _build_xgboost_model(n_samples=X.shape[0], seed=seed)
        model.fit(X_scaled, y)
        return {"scaler": scaler, "model": model, "model_type": model_type}

    if model_type == "svr":
        scaler = StandardScaler(with_std=True)
        X_scaled = scaler.fit_transform(X)
        model = _build_svr_model(n_samples=X.shape[0]).named_steps["svr"]
        model.fit(X_scaled, y)
        return {"scaler": scaler, "model": model, "model_type": model_type}

    raise ValueError(f"Unsupported model_type: {model_type}")


def predict(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = model_bundle["scaler"].transform(X)
    return model_bundle["model"].predict(X_scaled)


def train_and_predict_all_models(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict:
    _validate_weights(day_weights, feature_weights)

    ridge_predictions = run_pipeline(
        sentiment_path=sentiment_path,
        volatility_path=volatility_path,
        model_dir=model_dir,
        target_date=target_date,
        day_weights=day_weights,
        feature_weights=feature_weights,
    )

    sentiment_data = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    X, y = build_dataset(sentiment_data, volatility_data, target_date, day_weights, feature_weights)
    if X.shape[0] < 2:
        raise ValueError(f"Only {X.shape[0]} training samples - need at least 2.")

    X_pred, pred_tickers = build_prediction_rows(
        sentiment_data, volatility_data, target_date, day_weights, feature_weights
    )
    if len(pred_tickers) == 0:
        return {"ridge": ridge_predictions, "xgboost": {}, "svr": {}}

    xgb_bundle = train_model("xgboost", X, y, seed=42)
    _save_model(xgb_bundle, model_dir, "social_xgboost_model.pkl")
    xgb_preds = predict(xgb_bundle, X_pred)

    svr_bundle = train_model("svr", X, y, seed=42)
    _save_model(svr_bundle, model_dir, "social_svr_model.pkl")
    svr_preds = predict(svr_bundle, X_pred)

    return {
        "ridge": ridge_predictions,
        "xgboost": dict(zip(pred_tickers, xgb_preds.tolist())),
        "svr": dict(zip(pred_tickers, svr_preds.tolist())),
    }


def evaluate_all_models(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict:
    _validate_weights(day_weights, feature_weights)
    print("\n--- Ridge ---")
    ridge_results = run_eval_pipeline(
        sentiment_path=sentiment_path,
        volatility_path=volatility_path,
        model_dir=model_dir,
        target_date=target_date,
        day_weights=day_weights,
        feature_weights=feature_weights,
    )

    sentiment_data = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    from pipelines.prediction.ridgeRegression.eval_social_pipeline import build_eval_rows

    X, y_true, tickers = build_eval_rows(
        sentiment_data, volatility_data, target_date, day_weights, feature_weights
    )
    if len(tickers) == 0:
        raise ValueError("No tickers had both a sentiment window and a volatility label.")
    y_true_arr = np.array(y_true)

    print("\n--- XGBoost ---")
    xgb_bundle = _load_model(model_dir, "social_xgboost_model.pkl")
    xgb_pred = predict(xgb_bundle, X)
    xgb_stats = compute_stats(y_true_arr, xgb_pred)
    print_stats(xgb_stats, target_date, day_weights, feature_weights)

    print("\n--- SVR ---")
    svr_bundle = _load_model(model_dir, "social_svr_model.pkl")
    svr_pred = predict(svr_bundle, X)
    svr_stats = compute_stats(y_true_arr, svr_pred)
    print_stats(svr_stats, target_date, day_weights, feature_weights)

    return {
        "ridge": ridge_results,
        "xgboost": {
            "predictions": dict(zip(tickers, xgb_pred.tolist())),
            "actuals": dict(zip(tickers, y_true)),
            "stats": xgb_stats,
        },
        "svr": {
            "predictions": dict(zip(tickers, svr_pred.tolist())),
            "actuals": dict(zip(tickers, y_true)),
            "stats": svr_stats,
        },
    }
