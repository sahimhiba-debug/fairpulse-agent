"""Tests d'ENDPOINTS via le TestClient FastAPI — « l'app démarre / un endpoint répond ».

Astuce : on instancie TestClient SANS le bloc `with`. Du coup le *lifespan* n'est PAS
déclenché -> pas de session MCP, pas de modèle chargé. On teste donc la couche HTTP
(routing + validation) de façon légère et déterministe, ce qui convient au CI (où il
n'y a ni poids ni données restreintes).

• GET /health      -> 200, ne dépend pas de l'état applicatif -> répond directement.
• POST /audit (mauvaise entrée) -> 422 : la validation Pydantic rejette AVANT le handler,
  donc pas besoin de l'état (file/worker). C'est exactement notre garde-fou.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)  # pas de `with` => lifespan non exécuté (ni MCP ni modèle)


def test_health_responds_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body              # cpu en CI/conteneur, mps en local


def test_audit_rejects_out_of_bounds_n_boot():
    r = client.post("/audit", json={"n_boot": 99})
    assert r.status_code == 422          # garde-fou ressources


def test_audit_rejects_unknown_model():
    r = client.post("/audit", json={"model_name": "GPT-PPG"})
    assert r.status_code == 422          # liste blanche de modèles
