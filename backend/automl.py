from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pycaret.classification import (
    compare_models as compare_classification_models,
    load_model as load_classification_model,
    pull as pull_classification_results,
    save_model as save_classification_model,
    setup as setup_classification,
)
from pycaret.regression import (
    compare_models as compare_regression_models,
    load_model as load_regression_model,
    pull as pull_regression_results,
    save_model as save_regression_model,
    setup as setup_regression,
)
from pycaret.time_series import (
    compare_models as compare_time_series_models,
    load_model as load_time_series_model,
    predict_model as predict_time_series_model,
    pull as pull_time_series_results,
    save_model as save_time_series_model,
    setup as setup_time_series,
)

from backend import data_store


@dataclass
class TrainingResult:
    model_id: str
    metrics: Dict[str, float]
    rank: int
    algorithm: str
    problem_type: str
    artifacts_path: Path


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = df.describe(include="all").fillna("").to_dict()
    missing = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    correlations = df.select_dtypes(include=["number"]).corr().fillna(0).to_dict()
    sample = df.head(10).to_dict(orient="records")
    return {
        "summary": summary,
        "missing": missing,
        "dtypes": dtypes,
        "correlations": correlations,
        "sample": sample,
    }


def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
    return normalized


def _save_model(
    problem_type: str,
    model: Any,
    artifacts_dir: Path,
) -> tuple[str, Path]:
    model_id = str(uuid.uuid4())
    model_path = artifacts_dir / model_id
    if problem_type == "classification":
        save_classification_model(model, str(model_path))
    elif problem_type == "regression":
        save_regression_model(model, str(model_path))
    else:
        save_time_series_model(model, str(model_path))
    return model_id, model_path.with_suffix(".pkl")


def train_classification(df: pd.DataFrame, target: str, artifacts_dir: Path) -> List[TrainingResult]:
    setup_classification(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )
    models = compare_classification_models(
        n_select=3,
        include=["xgboost", "lr", "rf"],
        sort="Accuracy",
        verbose=False
    )
    if not isinstance(models, list):
        models = [models]
    results_df = pull_classification_results()
    rows = results_df.head(len(models)).to_dict(orient="records") if not results_df.empty else []

    results: List[TrainingResult] = []
    for model, row in zip(models, rows):
        model_id, model_path = _save_model("classification", model, artifacts_dir)
        metrics = _normalize_metrics(row)
        algorithm = row.get("Model") if isinstance(row, dict) else type(model).__name__
        results.append(
            TrainingResult(
                model_id=model_id,
                metrics=metrics,
                rank=0,
                algorithm=algorithm,
                problem_type="classification",
                artifacts_path=model_path,
            )
        )
    return results


def train_regression(df: pd.DataFrame, target: str, artifacts_dir: Path) -> List[TrainingResult]:
    setup_regression(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )
    models = compare_regression_models(
        n_select=3,
        include=["lasso", "rf", "xgboost"],
        sort="RMSE",
        verbose=False
    )
    if not isinstance(models, list):
        models = [models]
    results_df = pull_regression_results()
    rows = results_df.head(len(models)).to_dict(orient="records") if not results_df.empty else []

    results: List[TrainingResult] = []
    for model, row in zip(models, rows):
        model_id, model_path = _save_model("regression", model, artifacts_dir)
        metrics = _normalize_metrics(row)
        algorithm = row.get("Model") if isinstance(row, dict) else type(model).__name__
        results.append(
            TrainingResult(
                model_id=model_id,
                metrics=metrics,
                rank=0,
                algorithm=algorithm,
                problem_type="regression",
                artifacts_path=model_path,
            )
        )
    return results


def train_time_series(
    df: pd.DataFrame,
    target: str,
    time_column: str | None,
    artifacts_dir: Path,
) -> List[TrainingResult]:
    if time_column and time_column in df.columns:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(
                df[time_column],
                dayfirst=True,
                errors="coerce"
            )
        df = df.sort_values(by=time_column)

        # FIXME: GENERALIZAR ISSO AQUI
        df = df.set_index(time_column)
        df = df.asfreq('D')
        df = df.fillna({"Temperature": 0})
        series = df[target]
    else:
        series = df[target]

    setup_time_series(
        data=series,
        fh=12,
        session_id=42,
        verbose=True,
    )
    models = compare_time_series_models(
        # include=["auto_arima", "xgboost_cds_dt", "lightgbm_cds_dt"],
        include=["xgboost_cds_dt"],
        n_select=1,
        sort="RMSE",
        verbose=True
    )
    if not isinstance(models, list):
        models = [models]
    results_df = pull_time_series_results()
    rows = results_df.head(len(models)).to_dict(orient="records") if not results_df.empty else []

    results: List[TrainingResult] = []
    for model, row in zip(models, rows):
        model_id, model_path = _save_model("time_series", model, artifacts_dir)
        metrics = _normalize_metrics(row)
        algorithm = row.get("Model") if isinstance(row, dict) else type(model).__name__
        results.append(
            TrainingResult(
                model_id=model_id,
                metrics=metrics,
                rank=0,
                algorithm=algorithm,
                problem_type="time_series",
                artifacts_path=model_path,
            )
        )
    return results


def _score_classification(metrics: Dict[str, float]) -> float:
    for key in ("Accuracy", "F1", "AUC"):
        if key in metrics:
            return metrics[key]
    return 0.0


def _score_regression(metrics: Dict[str, float]) -> float:
    for key in ("RMSE", "MAE"):
        if key in metrics:
            return -metrics[key]
    return 0.0


def rank_results(results: List[TrainingResult]) -> List[TrainingResult]:
    if not results:
        return results
    if results[0].problem_type == "classification":
        key = lambda r: _score_classification(r.metrics)
    else:
        key = lambda r: _score_regression(r.metrics)
    sorted_results = sorted(results, key=key, reverse=True)
    for idx, result in enumerate(sorted_results, start=1):
        result.rank = idx
    return sorted_results


def persist_results(
    dataset_id: str,
    results: List[TrainingResult],
    target: str,
    time_column: str | None,
) -> Dict[str, Any]:
    payloads = []
    for result in results:
        model_payload = {
            "model_id": result.model_id,
            "dataset_id": dataset_id,
            "problem_type": result.problem_type,
            "algorithm": result.algorithm,
            "metrics": result.metrics,
            "rank": result.rank,
            "artifact_path": str(result.artifacts_path),
            "target": target,
            "time_column": time_column,
        }
        data_store.register_model(result.model_id, model_payload)
        payloads.append(model_payload)
    return {"models": payloads, "best_model": payloads[0] if payloads else None}


def load_model(model_entry: Dict[str, Any]) -> Any:
    path = Path(model_entry["artifact_path"])
    model_base = str(path.with_suffix(""))
    if model_entry["problem_type"] == "classification":
        return load_classification_model(model_base)
    if model_entry["problem_type"] == "regression":
        return load_regression_model(model_base)
    return load_time_series_model(model_base)


def forecast_time_series(model_entry: Dict[str, Any], steps: int) -> List[float]:
    model = load_model(model_entry)
    predictions = predict_time_series_model(model, fh=steps)
    if isinstance(predictions, pd.Series):
        values = predictions.values
    elif isinstance(predictions, pd.DataFrame):
        if "y_pred" in predictions.columns:
            values = predictions["y_pred"].values
        else:
            values = predictions.iloc[:, 0].values
    else:
        values = list(predictions)
    return [float(value) for value in values]


def save_metrics_report(dataset_id: str, payload: Dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps({"dataset_id": dataset_id, **payload}, ensure_ascii=False, indent=2))
