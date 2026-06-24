# syntax=docker/dockerfile:1
#
# Image de l'API FairPulse (couche 4, palier 3).
#
# Concepts :
# • Une IMAGE = un modèle figé (système + dépendances + code) ; un CONTENEUR = une
#   instance qui tourne à partir d'une image. On build l'image une fois, on lance
#   autant de conteneurs qu'on veut.
# • Le Dockerfile décrit, couche par couche, comment construire l'image. Chaque
#   instruction (RUN/COPY) crée une couche mise en cache -> on met ce qui change
#   le moins (les deps) AVANT le code, pour ne pas tout réinstaller à chaque edit.
#
# ⚠️ CE QUI N'EST PAS DANS L'IMAGE (garde-fous) : les POIDS du modèle, les DONNÉES
#    OpenOximetry (restreintes) et le code `fairpulse` sont MONTÉS en lecture seule
#    au runtime (voir docker-compose.yml), jamais copiés ici. Aucun secret non plus :
#    les credentials Postgres arrivent par variables d'environnement.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# libsndfile1 : requis par soundfile (dépendance transitive de sentence-transformers).
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch CPU depuis l'index PyTorch CPU (sur Linux, le wheel par défaut est CUDA,
#    inutile ici : un conteneur Linux n'a de toute façon PAS accès au GPU Apple/MPS).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2) Le reste des dépendances (couche cachée tant que requirements.txt ne change pas).
COPY api/deploy/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 3) Le code de l'agent (contexte de build = dossier fairpulse_agent/).
#    `fairpulse` lui-même n'est PAS copié : il est monté read-only au runtime.
COPY . /app/fairpulse_agent

# `agent` et `mcp_server`/`rag`/`api` importables ; le bootstrap d'agent/__init__.py
# ajoutera /app/src (fairpulse monté) au sys.path à l'exécution.
ENV PYTHONPATH=/app/fairpulse_agent:/app/fairpulse_agent/src \
    PYTORCH_ENABLE_MPS_FALLBACK=1

EXPOSE 8000

# host 0.0.0.0 : écouter sur toutes les interfaces DU CONTENEUR (sinon injoignable
# depuis l'extérieur). La publication de port (compose) restreint à localhost de l'hôte.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
