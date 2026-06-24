"""Jobs d'audit : stockage EN MÉMOIRE (palier 1) + worker de file + observabilité.

Pourquoi un job de fond ? Un audit est LONG (inférence MPS + bootstrap + RAG). Si
l'endpoint HTTP exécutait tout avant de répondre, la connexion resterait bloquée
(timeouts proxy/client) et le client n'aurait aucune visibilité. À la place :
  POST /audit   -> crée un job (status=queued), le met dans une file, répond tout de suite.
  worker unique -> dépile et exécute UN audit à la fois (sérialise le GPU = garde-fou).
  GET /audit/id -> renvoie le statut courant, puis le résultat quand status=done.

PALIER 1 : le JobStore vit en mémoire (dict). On le perd au redémarrage — acceptable
ici ; au palier 2 on remplacera l'implémentation par Postgres SANS toucher aux
endpoints (même interface create/get/mark_*).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from api.observability import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobRecord:
    job_id: str
    status: str                       # queued | running | done | error
    params: dict
    timings: list = field(default_factory=list)   # [{"node":…, "elapsed_ms":…}]
    error: str | None = None
    report_markdown: str | None = None
    result: dict | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def as_status(self) -> dict:
        return asdict(self)


class JobStore:
    """Registre des jobs en mémoire. Interface volontairement minimale et stable
    (au palier 2, une implémentation Postgres exposera les mêmes méthodes)."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create(self, params: dict) -> JobRecord:
        job_id = uuid.uuid4().hex
        rec = JobRecord(job_id=job_id, status="queued", params=params)
        self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def _touch(self, rec: JobRecord) -> None:
        rec.updated_at = _now_iso()

    def mark_running(self, job_id: str) -> None:
        rec = self._jobs[job_id]
        rec.status = "running"
        self._touch(rec)

    def add_timing(self, job_id: str, node: str, elapsed_ms: float) -> None:
        rec = self._jobs[job_id]
        rec.timings.append({"node": node, "elapsed_ms": round(elapsed_ms, 1)})
        self._touch(rec)

    def mark_done(self, job_id: str, report_md: str, result: dict) -> None:
        rec = self._jobs[job_id]
        rec.status = "done"
        rec.report_markdown = report_md
        rec.result = result
        self._touch(rec)

    def mark_error(self, job_id: str, error: str) -> None:
        rec = self._jobs[job_id]
        rec.status = "error"
        rec.error = error
        self._touch(rec)


async def run_job(app_state, store: JobStore, job_id: str) -> None:
    """Exécute UN audit (le graphe RAG couche 3) via la session MCP persistante.

    Observabilité : on consomme le graphe en STREAMING (`astream` mode "updates"),
    ce qui émet chaque nœud à sa complétion -> on loggue son nom + sa durée et on
    les stocke dans le job. Le modèle n'est PAS rechargé : il vit dans le serveur
    MCP (session ouverte au démarrage de l'API), chargé une fois sur MPS.
    """
    rec = store.get(job_id)
    store.mark_running(job_id)
    logger.info(f"job={job_id} event=start params={rec.params}")

    # L'agent attend un device dans sa config ; "auto" -> MPS résolu côté serveur.
    initial_state = {"config": {**rec.params, "device": "auto"}, "log": []}
    run_config = {"configurable": {"session": app_state.session}}

    merged: dict = {}
    t_prev = time.perf_counter()
    try:
        async for chunk in app_state.graph.astream(
            initial_state, config=run_config, stream_mode="updates"
        ):
            # chunk = {nom_du_noeud: {champs mis à jour}}
            for node, update in chunk.items():
                now = time.perf_counter()
                dt_ms = (now - t_prev) * 1000.0
                t_prev = now
                store.add_timing(job_id, node, dt_ms)
                logger.info(f"job={job_id} node={node} elapsed_ms={dt_ms:.1f}")
                if isinstance(update, dict):
                    merged.update(update)   # reconstitue les pièces de l'état final

        # Résultat = les sorties JSON des outils MCP + le contexte RAG + le markdown.
        result = {
            "metrics": merged.get("metrics"),
            "calibration": merged.get("calibration"),
            "robustness": merged.get("robustness"),
            "context": merged.get("context"),
        }
        store.mark_done(job_id, report_md=merged.get("report_md", ""), result=result)
        logger.info(f"job={job_id} event=done n_timings={len(rec.timings)}")
    except Exception as e:  # noqa: BLE001  (on veut capturer tout échec d'audit)
        store.mark_error(job_id, f"{type(e).__name__}: {e}")
        logger.exception(f"job={job_id} event=error err={e}")
