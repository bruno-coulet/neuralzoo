"""
Cartouche module:
- Projet: Interface Streamlit pour inference CIFAR-10.
- Role: upload image, appel API FastAPI, affichage prediction.
- Fonctionnalité : Sélection dynamique du modèle exporté.
- Compatibilite: Docker compose.
"""

from __future__ import annotations

import os
import requests
import json
import streamlit as st
from PIL import Image
from pathlib import Path  # Import indispensable pour scanner les fichiers

API_URL = os.getenv("API_URL", "http://localhost:8000")

def call_predict_api(image_bytes: bytes, filename: str, model_name: str) -> dict:
    """Envoie l'image ET le nom du modèle sélectionné à l'API."""
    files = {"file": (filename, image_bytes, "image/png")}
    # Le nom du modèle est passé en paramètre de requête (query parameter)
    params = {"model_name": model_name}
    
    response = requests.post(
        f"{API_URL}/predict", 
        files=files, 
        params=params, 
        timeout=60
    )
    response.raise_for_status()
    return response.json()

def load_model_details(model_name: str) -> dict | None:
    """Charge le fichier .json associé au modèle s'il existe."""
    json_path = Path("exports") / model_name.replace(".keras", ".json")
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def main() -> None:
    """Point d'entree principal de l'interface Streamlit."""
    st.set_page_config(page_title="CIFAR-10 Demo", layout="centered")
    st.title("Classification d'image CIFAR-10")
    st.caption("Frontend Streamlit connecté à une API FastAPI")

    # --- 1. Configuration Sidebar : Choix du modèle ---
    st.sidebar.title("Configuration du Modèle")
    export_dir = Path("exports")
    
    # Modèle par défaut si aucun fichier n'est trouvé
    selected_model = "cifar10_cnn.keras"

    if export_dir.exists():
        # On liste les fichiers .keras présents dans le volume partagé
        available_models = [f.name for f in export_dir.glob("*.keras")]
        
        # On trie pour avoir le plus récent (date de modification) en haut de liste
        available_models.sort(key=lambda x: (export_dir / x).stat().st_mtime, reverse=True)
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Choisir le modèle d'inférence :",
                options=available_models,
                index=0
            )
            st.sidebar.info(f"Modèle actif : {selected_model}")
    else:
        st.sidebar.warning("Dossier /exports non trouvé. Utilisation du défaut.")

    # --- Affichage des détails du modèle dans la Sidebar ---
    details = load_model_details(selected_model)

    if details:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Performances (Entraînement)")
        
        # Affichage des métriques clés
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Précision", f"{details.get('val_accuracy', 0):.1%}")
        col2.metric("Loss", f"{details.get('val_loss', 0):.3f}")
        
        # Détails techniques dans un expander pour ne pas encombrer
        with st.sidebar.expander("Détails techniques"):
            st.write(f"**Architecture:** {details.get('architecture')}")
            st.write(f"**Époques:** {details.get('epochs_trained')}")
            st.write(f"**Augmentation:** {'Oui' if details.get('data_augmentation') else 'Non'}")
            st.write(f"**Dropout:** {details.get('dropout_rate')}")

    # --- 2. Interface de téléchargement ---
    uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Image chargée", use_container_width=True)

        # --- 3. Action de prédiction ---
        if st.button("Prédire", type="primary"):
            try:
                with st.spinner(f"Analyse avec {selected_model}..."):
                    result = call_predict_api(
                        uploaded.getvalue(), 
                        uploaded.name, 
                        selected_model
                    )
                
                st.success("Prédiction terminée")
                
                # Affichage des résultats principaux
                col1, col2 = st.columns(2)
                col1.metric("Classe", result['label'])
                col2.metric("Confiance", f"{result['score']:.2%}")
                
                st.write(f"*Moteur d'inférence : {result['backend']}*")

                # Affichage du Top 5
                st.subheader("Détails du classement (Top 5)")
                for item in result['top_k']:
                    score_pct = item['score'] * 100
                    st.write(f"**{item['label']}**")
                    st.progress(item['score'])
                    st.caption(f"Probabilité : {score_pct:.2f}%")
                    
            except requests.RequestException as exc:
                st.error(f"Erreur de communication avec l'API : {exc}")
    else:
        st.info("Veuillez importer une image (PNG, JPG) pour lancer l'analyse.")

if __name__ == "__main__":
    main()