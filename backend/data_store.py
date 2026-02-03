from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
DATASET_DIR = STORAGE_DIR / "datasets"
MODEL_DIR = STORAGE_DIR / "models"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _metadata_path(dataset_id: str) -> Path:
    return DATASET_DIR / dataset_id / "metadata.json"


def _dataset_path(dataset_id: str) -> Path:
    return DATASET_DIR / dataset_id / "data.csv"


def create_dataset(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    dataset_folder = DATASET_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(_dataset_path(dataset_id), index=False)
    metadata = {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": len(df),
        "columns": df.columns.tolist(),
    }
    _metadata_path(dataset_id).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def load_dataset(dataset_id: str) -> pd.DataFrame:
    path = _dataset_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} not found")
    return pd.read_csv(path)


def save_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    df.to_csv(_dataset_path(dataset_id), index=False)
    metadata = load_metadata(dataset_id)
    metadata.update({"rows": len(df), "columns": df.columns.tolist()})
    _metadata_path(dataset_id).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def load_metadata(dataset_id: str) -> Dict[str, Any]:
    path = _metadata_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Metadata for dataset {dataset_id} not found")
    return json.loads(path.read_text())


def list_datasets() -> list[Dict[str, Any]]:
    datasets = []
    for folder in DATASET_DIR.iterdir():
        if folder.is_dir():
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                datasets.append(json.loads(meta_path.read_text()))
    return datasets


def registry_path() -> Path:
    return MODEL_DIR / "registry.json"


def deployments_path() -> Path:
    return MODEL_DIR / "deployments.json"


def load_registry() -> Dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text())


def save_registry(registry: Dict[str, Any]) -> None:
    registry_path().write_text(json.dumps(registry, ensure_ascii=False, indent=2))


def register_model(model_id: str, payload: Dict[str, Any]) -> None:
    registry = load_registry()
    registry["models"][model_id] = payload
    save_registry(registry)


def get_model_entry(model_id: str) -> Dict[str, Any]:
    registry = load_registry()
    model = registry["models"].get(model_id)
    if not model:
        raise FileNotFoundError(f"Model {model_id} not found")
    return model


def load_deployments() -> Dict[str, Any]:
    path = deployments_path()
    if not path.exists():
        return {"deployments": {}}
    return json.loads(path.read_text())


def save_deployments(deployments: Dict[str, Any]) -> None:
    deployments_path().write_text(json.dumps(deployments, ensure_ascii=False, indent=2))


def register_deployment(name: str, payload: Dict[str, Any]) -> None:
    deployments = load_deployments()
    deployments["deployments"][name] = payload
    save_deployments(deployments)


def get_deployment(name: str) -> Dict[str, Any]:
    deployments = load_deployments()
    deployment = deployments["deployments"].get(name)
    if not deployment:
        raise FileNotFoundError(f"Deployment {name} not found")
    return deployment
