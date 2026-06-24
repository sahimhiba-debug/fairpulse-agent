"""Schémas Pydantic — la VALIDATION des entrées est un garde-fou de sécurité.

Pourquoi c'est un garde-fou : FastAPI valide CHAQUE requête contre ces modèles
AVANT d'exécuter le code. Une entrée hors-bornes (n_boot=10⁹, modèle inconnu,
type erroné) est rejetée automatiquement avec un 422 explicite — le moteur d'audit
ne voit jamais de données aberrantes. `Literal[...]` restreint à une liste blanche ;
`Field(ge=, le=)` borne les valeurs (protège la mémoire/CPU/GPU).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AuditRequest(BaseModel):
    """Paramètres d'un audit. Tout est borné -> impossible de saturer la machine."""

    model_name: Literal["PaPaGei-S", "Pulse-PPG"] = "PaPaGei-S"  # liste blanche
    target: Literal["spo2", "sao2"] = "spo2"
    n_boot: int = Field(500, ge=100, le=2000, description="Rééchantillonnages bootstrap (borné).")
    max_patients_per_group: int = Field(6, ge=1, le=20)
    max_windows_per_encounter: int = Field(8, ge=1, le=40)
    seed: int = Field(0, ge=0, le=2_147_483_647)


class JobCreated(BaseModel):
    job_id: str
    status: str


class NodeTiming(BaseModel):
    node: str
    elapsed_ms: float


class JobStatus(BaseModel):
    job_id: str
    status: str                              # queued | running | done | error
    params: dict[str, Any]
    timings: list[NodeTiming] = []           # observabilité : durée par nœud
    error: Optional[str] = None
    report_markdown: Optional[str] = None    # rapport contextualisé (si done)
    result: Optional[dict[str, Any]] = None  # métriques + calibration + robustesse + contexte RAG
    created_at: str
    updated_at: str
