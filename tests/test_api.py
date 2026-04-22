"""Tests API FastAPI pour endpoint de sante et prediction image."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.app import app


def _build_test_png_bytes() -> bytes:
    """Construit une image PNG RGB en memoire pour les tests d'API."""
    image = Image.new("RGB", (32, 32), color=(120, 140, 200))
    payload = BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def test_health_endpoint() -> None:
    """Valide que l'endpoint /health renvoie un statut ok."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint_with_image() -> None:
    """Valide que l'endpoint /predict renvoie le schema attendu."""
    image_bytes = _build_test_png_bytes()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", image_bytes, "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert "label" in body
        assert "score" in body
        assert "backend" in body
        assert isinstance(body.get("top_k"), list)
        assert len(body["top_k"]) > 0
