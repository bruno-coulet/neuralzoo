"""
- Projet: Service FastAPI d'inference CIFAR-10.
- Role: couche HTTP (endpoints, schemas de reponse, cycle de vie du predictor).
- Endpoints: /health, /predict, /dataset/download.
- Integration: consomme src.model et alimente ui/app.py.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from pydantic import BaseModel

from src.model import CifarPredictor, download_cifar10_dataset, preprocess_image_bytes

# --- Schémas de réponse Pydantic ---

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

# --- Gestion du Predictor et du Cache ---

# On utilise un cache pour éviter de recharger le fichier .keras à chaque requête
_PREDICTORS_CACHE: Dict[str, CifarPredictor] = {}

def _get_or_create_predictor(model_name: str = None) -> CifarPredictor:
    """Récupère un prédicteur du cache ou en crée un nouveau selon le modèle choisi."""
    export_dir = Path("exports")
    
    # 1. Déterminer quel fichier charger
    if model_name:
        model_path = export_dir / model_name
    else:
        # Par défaut : chercher le fichier le plus récent dans /exports
        models = list(export_dir.glob("*.keras"))
        if not models:
            # Fallback ultime sur la variable d'environnement ou le nom standard
            model_path = Path(os.getenv("MODEL_PATH", "exports/cifar10_cnn.keras"))
        else:
            model_path = max(models, key=lambda p: p.stat().st_mtime)

    # 2. Gestion du cache
    cache_key = model_path.name
    if cache_key not in _PREDICTORS_CACHE:
        if not model_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Le fichier modèle {cache_key} est introuvable dans le dossier exports."
            )
        
        # Initialisation du prédicteur (chargement du modèle TensorFlow)
        _PREDICTORS_CACHE[cache_key] = CifarPredictor(model_path=model_path)
    
    return _PREDICTORS_CACHE[cache_key]

@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Gère le cycle de vie de l'API au démarrage et à la fermeture."""
    # Pré-chargement du modèle par défaut au démarrage
    try:
        _get_or_create_predictor()
    except Exception:
        pass 
    yield
    # Nettoyage du cache à la fermeture
    _PREDICTORS_CACHE.clear()

# --- Initialisation de l'application FastAPI ---

app = FastAPI(
    title="CIFAR10 Inference API", 
    version="1.1.0", 
    lifespan=lifespan
)

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de santé pour vérifier que l'API répond."""
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(None, description="Nom du fichier .keras à utiliser")
) -> PredictResponse:
    """Reçoit une image et renvoie la prédiction du modèle sélectionné."""
    
    # Vérification du type de fichier
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    # Lecture du contenu
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")

    # Récupération du prédicteur (via cache ou nouveau chargement)
    try:
        predictor = _get_or_create_predictor(model_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {str(e)}")

    # Prétraitement et Inférence
    image_batch = preprocess_image_bytes(payload)
    result = predictor.predict(image_batch=image_batch, top_k=5)
    
    return PredictResponse(**result)

@app.post("/dataset/download", response_model=DatasetDownloadResponse)
def download_dataset() -> DatasetDownloadResponse:
    """Télécharge le dataset CIFAR-10 en local."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    extracted_to = download_cifar10_dataset(data_dir)
    return DatasetDownloadResponse(extracted_to=str(extracted_to))