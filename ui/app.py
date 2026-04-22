"""
Cartouche module:
- Projet: Interface Streamlit pour inference CIFAR-10.
- Role: upload image, appel API FastAPI, affichage prediction.
- Compatibilite: Docker compose.
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("API_URL", "http://localhost:8000")


def call_predict_api(image_bytes: bytes, filename: str) -> dict:
    """Envoie une image a l'API et retourne la prediction JSON."""
    files = {"file": (filename, image_bytes, "image/png")}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Point d'entree principal de l'interface Streamlit."""
    st.set_page_config(page_title="CIFAR-10 Demo", layout="centered")
    st.title("Classification d'image CIFAR-10")
    st.caption("Frontend Streamlit connecte a une API FastAPI")

    uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Importer une image pour lancer une prediction.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image chargee", use_container_width=True)

    if st.button("Predire", type="primary"):
        try:
            result = call_predict_api(uploaded.getvalue(), uploaded.name)
        except requests.RequestException as exc:
            st.error(f"Erreur API: {exc}")
            return

        st.success("Prediction terminee")
        st.write(f"Classe predite: {result['label']}")
        st.write(f"Score: {result['score']:.4f}")
        st.write(f"Backend: {result['backend']}")

        st.subheader("Top 5")
        for item in result["top_k"]:
            st.write(f"- {item['label']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()
