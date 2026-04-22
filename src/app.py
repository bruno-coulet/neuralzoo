"""
Cartouche module:
- Projet: Service FastAPI d'inference CIFAR-10.
- Endpoints: /health, /predict, /dataset/download.
- Compatibilite: Docker + Streamlit frontend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.model import CifarPredictor, download_cifar10_dataset, preprocess_image_bytes


class TopKItem(BaseModel):
    """Element du classement top-k."""

    label: str
    score: float


class PredictResponse(BaseModel):
    """Schema de reponse de prediction."""

    label: str
    score: float
    backend: str
    top_k: list[TopKItem]


class HealthResponse(BaseModel):
    """Schema de reponse de sante."""

    status: str


class DatasetDownloadResponse(BaseModel):
    """Schema de reponse telechargement dataset."""

    extracted_to: str


def _build_predictor() -> CifarPredictor:
    """Construit l'instance de prediction a partir des variables d'environnement."""
    model_path = Path(os.getenv("MODEL_PATH", "exports/cifar10_cnn.keras"))
    auto_download = os.getenv("AUTO_DOWNLOAD_CIFAR", "false").lower() == "true"
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    return CifarPredictor(
        model_path=model_path,
        auto_download_cifar=auto_download,
        data_dir=data_dir,
    )


def _get_predictor() -> CifarPredictor:
    """Retourne le predictor present dans l'etat FastAPI, ou le cree si absent."""
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        predictor = _build_predictor()
        app.state.predictor = predictor
    return predictor


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Gere le cycle de vie de l'API (initialisation et nettoyage)."""
    app_instance.state.predictor = _build_predictor()
    yield


app = FastAPI(title="CIFAR10 Inference API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de sante pour CI/CD et supervision."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Reoit une image puis renvoie prediction CIFAR-10 et top-k."""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Fichier vide")

    image_batch = preprocess_image_bytes(payload)
    result = _get_predictor().predict(image_batch=image_batch, top_k=5)
    return PredictResponse(**result)


@app.post("/dataset/download", response_model=DatasetDownloadResponse)
def download_dataset() -> DatasetDownloadResponse:
    """Telecharge CIFAR-10 localement pour eviter des retelechargements repetes."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    extracted_to = download_cifar10_dataset(data_dir)
    return DatasetDownloadResponse(extracted_to=str(extracted_to))
