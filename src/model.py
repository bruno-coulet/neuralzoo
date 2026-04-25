"""
- Projet: Classification d'images (CIFAR-10) avec API d'inference.
- Role: coeur metier du modele (chargement, classes, preprocessing, prediction, export).
- Integration: utilise par src.app (FastAPI) et notebooks/eda.ipynb.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Tentative d'import de l'heuristique (fallback si TF n'est pas là)
try:
    from src.heuristic import predict_cifar10_heuristic_logits
except ImportError:
    # Fallback minimal si le fichier n'existe pas lors de la reconstruction
    def predict_cifar10_heuristic_logits(x): return np.zeros((len(x), 10))

try:
    import tensorflow as tf
except ImportError:
    tf = None


# --- Constantes ---

CLASS_NAMES_CIFAR10: list[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CLASS_NAMES_ANIMALS: list[str] = [
    "bird", "cat", "deer", "dog", "frog", "horse",
]

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# --- Fonctions utilitaires ---

def save_trained_model(model: Any, export_path: Path | str) -> None:
    """Sauvegarde un modèle Keras au format .keras."""
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def download_cifar10_dataset(target_dir: Path | str) -> Path:
    """Télécharge et extrait CIFAR-10 dans le dossier cible."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = target_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        print(f"Téléchargement de CIFAR-10 depuis {CIFAR_URL}...")
        urllib.request.urlretrieve(CIFAR_URL, archive_path)
    
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    
    return target_dir / "cifar-10-batches-py"


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Transforme les bytes d'une image en batch prêt pour le modèle (1, 32, 32, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32))
    # Normalisation : mise à l'échelle [0, 1]
    img_array = np.array(img).astype("float32") / 255.0
    # Ajout de la dimension batch
    return np.expand_dims(img_array, axis=0)


# --- Classe de prédiction ---

class CifarPredictor:
    """Gère le chargement du modèle et l'exécution de l'inférence."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        
        # Par défaut, si on utilise un modèle filtré "animaux", on charge ces noms
        # Sinon on charge les 10 noms CIFAR-10 par défaut.
        self.class_names = class_names or CLASS_NAMES_ANIMALS
        self.model = None
        self.backend = "Heuristic (Fallback)"

        if tf is not None and self.model_path and self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(self.model_path))
                self.backend = f"TensorFlow ({self.model_path.name})"
                
                # Ajustement dynamique des classes si le modèle a 10 sorties
                if self.model.output_shape[-1] == 10:
                    self.class_names = CLASS_NAMES_CIFAR10
            except Exception as e:
                print(f"Erreur chargement modèle : {e}")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, image_batch: np.ndarray, top_k: int = 5) -> dict:
        """Exécute la prédiction et formate le résultat."""
        if self.model is not None:
            probs = self.model.predict(image_batch, verbose=0)
        else:
            # Fallback heuristique si pas de modèle
            logits = predict_cifar10_heuristic_logits(image_batch)
            probs = self._softmax(logits)

        # Extraction des scores du premier (et seul) élément du batch
        scores = probs[0]
        order = np.argsort(scores)[::-1]
        
        # Sécurité sur le nombre de classes
        top_k_safe = min(top_k, len(self.class_names))
        top_idx = order[:top_k_safe]
        best_idx = int(top_idx[0])

        return {
            "label": self.class_names[best_idx],
            "score": float(scores[best_idx]),
            "backend": self.backend,
            "top_k": [
                {
                    "label": self.class_names[int(i)], 
                    "score": float(scores[int(i)])
                }
                for i in top_idx
            ],
        }

# --- CLI ---

def _main() -> None:
    parser = argparse.ArgumentParser(description="Utilitaires CIFAR-10")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    if args.download_only:
        download_cifar10_dataset(args.data_dir)

if __name__ == "__main__":
    _main()