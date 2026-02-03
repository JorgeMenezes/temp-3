from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import automl, data_store

app = FastAPI(title="AutoML Hackathon API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransformRequest(BaseModel):
    drop_columns: Optional[List[str]] = None
    fillna: Optional[Dict[str, Any]] = None
    add_columns: Optional[List[Dict[str, str]]] = None


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    problem_type: str = Field(..., pattern="^(classification|regression|time_series)$")
    time_column: Optional[str] = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class ForecastRequest(BaseModel):
    steps: int = 12


class BatchPredictRequest(BaseModel):
    dataset_id: str


class DeployRequest(BaseModel):
    model_id: str
    name: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "dataset"
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use CSV ou Excel.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {exc}") from exc
    return data_store.create_dataset(df, filename)


@app.get("/datasets")
async def list_datasets() -> List[Dict[str, Any]]:
    return data_store.list_datasets()


@app.get("/datasets/{dataset_id}/summary")
async def dataset_summary(dataset_id: str) -> Dict[str, Any]:
    try:
        df = data_store.load_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return automl.dataset_summary(df)


@app.get("/datasets/{dataset_id}/preview")
async def dataset_preview(dataset_id: str, rows: int = 10) -> List[Dict[str, Any]]:
    try:
        df = data_store.load_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return df.head(rows).to_dict(orient="records")


@app.post("/datasets/{dataset_id}/transform")
async def transform_dataset(dataset_id: str, payload: TransformRequest) -> Dict[str, Any]:
    try:
        df = data_store.load_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.drop_columns:
        df = df.drop(columns=payload.drop_columns, errors="ignore")

    if payload.fillna:
        df = df.fillna(payload.fillna)

    if payload.add_columns:
        for column in payload.add_columns:
            name = column.get("name")
            expression = column.get("expression")
            if not name or not expression:
                raise HTTPException(status_code=400, detail="add_columns requer name e expression")
            try:
                df[name] = df.eval(expression)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Erro na expressão {name}: {exc}") from exc

    data_store.save_dataset(dataset_id, df)
    return data_store.load_metadata(dataset_id)


@app.post("/train")
async def train_models(payload: TrainRequest) -> Dict[str, Any]:
    try:
        df = data_store.load_dataset(payload.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.target not in df.columns:
        raise HTTPException(status_code=400, detail="Coluna target não encontrada")

    artifacts_dir = data_store.MODEL_DIR
    if payload.problem_type == "classification":
        results = automl.train_classification(df, payload.target, artifacts_dir)
    elif payload.problem_type == "regression":
        results = automl.train_regression(df, payload.target, artifacts_dir)
    else:
        results = automl.train_time_series(df, payload.target, payload.time_column, artifacts_dir)

    ranked = automl.rank_results(results)
    response = automl.persist_results(payload.dataset_id, ranked, payload.target, payload.time_column)
    report_path = data_store.MODEL_DIR / f"report-{payload.dataset_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    automl.save_metrics_report(payload.dataset_id, response, report_path)
    return response


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    return data_store.load_registry()


@app.get("/models/{model_id}")
async def model_detail(model_id: str) -> Dict[str, Any]:
    try:
        return data_store.get_model_entry(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/models/{model_id}/predict")
async def predict(model_id: str, payload: PredictRequest) -> Dict[str, Any]:
    try:
        model_entry = data_store.get_model_entry(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    model = automl.load_model(model_entry)
    if model_entry["problem_type"] == "time_series":
        raise HTTPException(status_code=400, detail="Use /models/{model_id}/forecast para séries temporais")

    df = pd.DataFrame(payload.records)
    try:
        predictions = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao predizer: {exc}") from exc
    return {"predictions": predictions.tolist()}


@app.post("/models/{model_id}/forecast")
async def forecast(model_id: str, payload: ForecastRequest) -> Dict[str, Any]:
    try:
        model_entry = data_store.get_model_entry(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if model_entry["problem_type"] != "time_series":
        raise HTTPException(status_code=400, detail="Forecast apenas para séries temporais")

    forecast_values = automl.forecast_time_series(model_entry, payload.steps)
    return {"forecast": forecast_values}


@app.get("/deployments")
async def list_deployments() -> Dict[str, Any]:
    return data_store.load_deployments()


@app.post("/deployments")
async def create_deployment(payload: DeployRequest) -> Dict[str, Any]:
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Nome do deploy é obrigatório")
    if not payload.name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Nome do deploy deve conter apenas letras, números, - ou _")

    try:
        model_entry = data_store.get_model_entry(payload.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if model_entry["problem_type"] == "time_series":
        raise HTTPException(status_code=400, detail="Deploy não suporta séries temporais")

    deployments = data_store.load_deployments()
    if payload.name in deployments.get("deployments", {}):
        raise HTTPException(status_code=409, detail="Já existe um deploy com esse nome")

    deployment = {
        "name": payload.name,
        "model_id": payload.model_id,
        "target": model_entry.get("target"),
        "problem_type": model_entry.get("problem_type"),
        "algorithm": model_entry.get("algorithm"),
    }
    data_store.register_deployment(payload.name, deployment)
    return deployment


@app.get("/deployments/{deployment_name}")
async def get_deployment(deployment_name: str) -> Dict[str, Any]:
    try:
        return data_store.get_deployment(deployment_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/deployments/{deployment_name}/predict")
async def predict_deployment(deployment_name: str, payload: PredictRequest) -> Dict[str, Any]:
    try:
        deployment = data_store.get_deployment(deployment_name)
        model_entry = data_store.get_model_entry(deployment["model_id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if model_entry["problem_type"] == "time_series":
        raise HTTPException(status_code=400, detail="Deploy não suporta séries temporais")

    model = automl.load_model(model_entry)
    df = pd.DataFrame(payload.records)
    if model_entry.get("target") in df.columns:
        df = df.drop(columns=[model_entry["target"]])
    try:
        predictions = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao predizer: {exc}") from exc
    return {"predictions": predictions.tolist()}


@app.post("/deployments/{deployment_name}/predict-file")
async def predict_deployment_file(deployment_name: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        deployment = data_store.get_deployment(deployment_name)
        model_entry = data_store.get_model_entry(deployment["model_id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if model_entry["problem_type"] == "time_series":
        raise HTTPException(status_code=400, detail="Deploy não suporta séries temporais")

    filename = file.filename or "dataset"
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use CSV ou Excel.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {exc}") from exc

    if model_entry.get("target") in df.columns:
        df = df.drop(columns=[model_entry["target"]])

    model = automl.load_model(model_entry)
    try:
        predictions = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao predizer: {exc}") from exc

    output_df = df.copy()
    output_df["prediction"] = predictions
    output_path = data_store.MODEL_DIR / f"deploy-{deployment_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    output_df.to_csv(output_path, index=False)
    return {
        "output_file": str(output_path),
        "output_filename": output_path.name,
        "download_url": f"/files/{output_path.name}",
        "rows": len(output_df),
    }


@app.get("/files/{file_name}")
async def download_file(file_name: str) -> FileResponse:
    if "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido")
    file_path = data_store.MODEL_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(path=file_path, filename=file_name)


@app.post("/batch/predict")
async def batch_predict(payload: BatchPredictRequest, model_id: str) -> Dict[str, Any]:
    try:
        model_entry = data_store.get_model_entry(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if model_entry["problem_type"] == "time_series":
        raise HTTPException(status_code=400, detail="Batch predict não suporta séries temporais")

    df = data_store.load_dataset(payload.dataset_id)
    if model_entry.get("target") in df.columns:
        df = df.drop(columns=[model_entry["target"]])

    model = automl.load_model(model_entry)
    predictions = model.predict(df)
    output_path = data_store.MODEL_DIR / f"batch-{payload.dataset_id}-{model_id}.csv"
    result_df = df.copy()
    result_df["prediction"] = predictions
    result_df.to_csv(output_path, index=False)
    return {"output_file": str(output_path), "rows": len(result_df)}
