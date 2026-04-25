# NeuralZOO

Projet de classification d'images CIFAR-10 avec comparaison MLP vs CNN, puis mise en production d'un service d'inference avec FastAPI, Streamlit et Docker.

## Quickstart

1. Creer et activer l'environnement Conda.
2. Installer les dependances API/UI.
3. Lancer les tests.
4. Demarrer la stack Docker.

```bash
conda create -n neuralzoo python=3.11 -y
conda activate neuralzoo
conda install -c conda-forge tensorflow scikit-learn pandas -y
pip install -r requirements-api.txt -r requirements-ui.txt
pytest -q
docker compose up --build
```

Source dataset utilisee:
- CIFAR-10 python version (https://www.cs.toronto.edu/~kriz/cifar.html)

## 1) Objectifs

- Construire une baseline MLP et un modele CNN sur CIFAR-10.
- Structurer un notebook compatible Google Colab.
- Exposer l'inference via API FastAPI.
- Ajouter une interface Streamlit.
- Conteneuriser et automatiser des verifications CI.

## 2) Donnees

Source officielle CIFAR-10:
- https://www.cs.toronto.edu/~kriz/cifar.html

Version utilisee dans le projet:
- CIFAR-10 python version

Caracteristiques:
- 60 000 images RGB 32x32
- 10 classes
- 50 000 train / 10 000 test

Le notebook telecharge directement l'archive officielle CIFAR-10 python version et charge le format Python (pickle).

Sous-ensemble actuellement utilise pour l'entrainement du modele exporte:
- classes animales uniquement: bird, cat, deer, dog, frog, horse
- train: 30 000 images (puis split train/val)
- test: 6 000 images

## 3) Etat actuel du depot

Arborescence principale:

- notebooks/: notebooks d'experimentation (dont notebook Colab-ready)
- src/: API FastAPI et logique de prediction
- ui/: application Streamlit
- tests/: tests API
- Dockerfile.api: image API
- Dockerfile.ui: image UI
- docker-compose.yml: orchestration locale
- requirements-api.txt: dependances API/tests
- requirements-ui.txt: dependances UI
- environment.yml: environnement Conda local
- .github/workflows/ci.yml: pipeline CI

## 4) Environnement local (Conda)

Le projet utilise un environnement Conda dedie nomme neuralzoo.

Creation:

```bash
conda create -n neuralzoo python=3.11 -y
conda activate neuralzoo
conda install -c conda-forge tensorflow scikit-learn pandas -y
pip install -r requirements-api.txt -r requirements-ui.txt
```

Tests:

```bash
pytest -q
```

Export minimal reproductible:

```bash
conda env export -n neuralzoo --from-history > environment.yml
```

## 5) Notebook Colab

Notebook principal:
- notebooks/eda.ipynb

Contenu:
- setup Colab (verification Python/TensorFlow/GPU)
- telechargement CIFAR-10 depuis Toronto
- chargement, filtrage des classes animales, puis pretraitement
- entrainement MLP
- entrainement CNN
- comparaison des performances
- courbes d'apprentissage
- matrice de confusion

## 6) API FastAPI

Fichier principal:
- src/app.py

Endpoints:
- GET /health: etat de service
- POST /predict: prediction sur image envoyee
- POST /dataset/download: telechargement CIFAR-10 en local

Comportement:
- si un modele exporte existe dans exports/cifar10_cnn.keras, l'API l'utilise
- sinon, un fallback heuristique permet de garder le service operationnel

Etat backend valide en Docker:
- backend tensorflow actif quand le fichier exports/cifar10_cnn.keras est present
- fallback heuristique uniquement si modele absent ou non chargeable

## 7) Interface Streamlit

Fichier principal:
- ui/app.py

Fonctionnalites:
- upload d'image
- appel de l'API /predict
- affichage classe predite, score, backend, top-5

## 8) Docker et Docker Compose

Validation config:

```bash
docker compose config
```

Build + run:

```bash
docker compose up --build
```

Acces:
- API: http://localhost:8000/health
- UI: http://localhost:8501

Prechargement dataset (profil optionnel):

```bash
docker compose --profile dataset up dataset
```

Cette commande telecharge CIFAR-10 dans le dossier data/ pour accelerer les executions suivantes.

## 9) CI

Workflow:
- .github/workflows/ci.yml

Etapes:
- installation des dependances API/tests
- execution de pytest
- build des images Docker API et UI

## 10) Remarques

- Le 404 sur /favicon.ico dans les logs API est non bloquant.
- Le projet est valide localement (tests OK, compose OK, services demarres).
- Les predictions actuelles sont fonctionnelles mais qualitativement insuffisantes pour une mise en production.
- Le modele exporte actuel est un CNN entraine sur 6 classes animales (pas sur 10 classes CIFAR-10).

## 11) Suite possible

- Ameliorer la qualite du CNN (data augmentation, regularisation, scheduler LR, tuning hyperparametres).
- Re-entrainer et exporter une version plus robuste du modele.
- Ajouter des metriques de suivi (accuracy macro, f1 macro) et seuils de validation avant export.
- Ajouter des tests d'integration UI->API.
- Ajouter une etape de publication des images Docker en CI.
