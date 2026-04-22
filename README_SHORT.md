# NeuralZOO - README court

## Resume
NeuralZOO compare un MLP et un CNN sur CIFAR-10, puis expose l'inference via FastAPI + Streamlit, avec execution locale via Docker Compose.

## Source des donnees
- https://www.cs.toronto.edu/~kriz/cifar.html

## Composants
- Notebook Colab-ready: notebooks/eda.ipynb
- API inference: src/app.py
- Logique modele/dataset: src/model.py
- Interface utilisateur: ui/app.py
- Tests: tests/test_api.py

## Environnement local
Le projet utilise Conda.

```bash
conda create -n neuralzoo python=3.11 -y
conda activate neuralzoo
conda install -c conda-forge tensorflow scikit-learn pandas -y
pip install -r requirements-api.txt -r requirements-ui.txt
pytest -q
```

## Lancement Docker

```bash
docker compose config
docker compose up --build
```

Acces:
- API: http://localhost:8000/health
- UI: http://localhost:8501

Prechargement dataset (optionnel):

```bash
docker compose --profile dataset up dataset
```

## CI
Le workflow .github/workflows/ci.yml execute:
- tests pytest
- build image API
- build image UI

## Etat actuel
- Tests API: OK
- API + UI via Docker: OK
