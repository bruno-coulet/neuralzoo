"""
- Projet: Classification CIFAR-10 (MLP/CNN) avec service d'inference.
- Role: chargement modele, pretraitement image, prediction, telechargement dataset.
- Compatibilite: Python 3.10+.
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

from src.heuristic import predict_cifar10_heuristic_logits

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


CLASS_NAMES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_cifar10_dataset(data_dir: Path) -> Path:
    """Telecharge CIFAR-10 (source Toronto) puis extrait les fichiers Python."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "cifar-10-python.tar.gz"
    extract_dir = data_dir / "cifar-10-batches-py"

    if not archive_path.exists():
        urllib.request.urlretrieve(CIFAR_URL, archive_path)

    if not extract_dir.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    return extract_dir


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode une image, la redimensionne en 32x32, puis normalise dans [0,1]."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


class CifarPredictor:
    """Service de prediction CIFAR-10 avec backend TensorFlow ou fallback heuristique."""

    def __init__(
        self,
        model_path: Path = Path("exports/cifar10_cnn.keras"),
        auto_download_cifar: bool = False,
        data_dir: Path = Path("data"),
    ) -> None:
        self.model_path = model_path
        self.backend = "heuristic"
        self.model: Any | None = None

        if auto_download_cifar:
            download_cifar10_dataset(data_dir)

        if tf is not None and model_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            self.backend = "tensorflow"

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Calcule softmax de maniere numeriquement stable."""
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict(self, image_batch: np.ndarray, top_k: int = 5) -> dict[str, Any]:
        """Retourne prediction principale et top-k pour une image pretraitee."""
        if self.model is not None:
            probs = self.model.predict(image_batch, verbose=0)
        else:
            logits = predict_cifar10_heuristic_logits(image_batch)
            probs = self._softmax(logits)

        scores = probs[0]
        order = np.argsort(scores)[::-1]
        top_idx = order[:top_k]
        best_idx = int(top_idx[0])

        return {
            "label": CLASS_NAMES[best_idx],
            "score": float(scores[best_idx]),
            "backend": self.backend,
            "top_k": [
                {"label": CLASS_NAMES[int(i)], "score": float(scores[int(i)])}
                for i in top_idx
            ],
        }


def _main() -> None:
    """Point d'entree CLI pour precharger le dataset CIFAR-10."""
    parser = argparse.ArgumentParser(description="Utilitaires modele CIFAR-10")
    parser.add_argument("--download-only", action="store_true", help="Telecharge CIFAR-10")
    parser.add_argument("--data-dir", default="data", help="Dossier de destination")
    args = parser.parse_args()

    if args.download_only:
        extract_dir = download_cifar10_dataset(Path(args.data_dir))
        print(f"Dataset pret dans: {extract_dir}")


if __name__ == "__main__":
    _main()
