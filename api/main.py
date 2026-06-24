"""Application FastAPI — PALIER 1.

Points clés :
• LIFESPAN : au démarrage, on ouvre UNE session MCP (le serveur MCP charge le
  modèle PPG une seule fois, sur MPS) et on la garde ouverte pour toute la vie de
  l'API. Donc le modèle n'est JAMAIS rechargé par requête. On démarre aussi le
  worker de fond qui consomme la file de jobs.
• ASYNC : POST /audit met un job en file et répond tout de suite (202). Le worker
  exécute l'audit en arrière-plan. GET /audit/{id} interroge le statut/résultat.
• GUARDRAILS : validation Pydantic (schemas.py) + file bornée (429 si pleine).
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI, HTTPException, Response
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agent import config as agent_config
from agent.mcp_client import server_params
from agent.mcp_graph_rag import build_rag_graph
from api.jobs import JobStore, run_job
from api.observability import logger
from api.schemas import AuditRequest, JobCreated, JobStatus
from api.settings import settings

# Emplacement de l'index RAG (pour le diagnostic /health).
INDEX_DIR = agent_config.REPORTS_DIR.parent / "rag" / "index"


async def _worker_loop(app: FastAPI) -> None:
    """Worker UNIQUE : dépile les job_id et exécute un audit à la fois.

    Un seul worker => l'inférence MPS est sérialisée (deux audits ne se battent
    pas pour le GPU). C'est notre garde-fou ressources côté exécution.
    """
    queue: asyncio.Queue = app.state.queue
    while True:
        job_id = await queue.get()
        try:
            await run_job(app.state, app.state.store, job_id)
        finally:
            queue.task_done()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Démarrage/arrêt : ouvre la session MCP persistante + lance le worker."""
    logger.info("event=startup msg=ouverture_session_MCP")

    # Choix du store par ENV (inversion de dépendance) : DATABASE_URL défini ->
    # persistance PostgreSQL ; sinon -> mémoire (palier 1). On ne logge JAMAIS
    # l'URL complète (elle contient le mot de passe) : seulement le schéma.
    if settings.database_url:
        from api.db import SqlJobStore

        app.state.store = SqlJobStore(settings.database_url)
        logger.info(f"event=store backend=sql scheme={settings.database_url.split('://', 1)[0]}")
    else:
        app.state.store = JobStore()
        logger.info("event=store backend=memory")

    app.state.queue = asyncio.Queue(maxsize=settings.max_queue)

    # stdio_client lance le serveur MCP en sous-processus ; il reste vivant tant
    # que ce bloc `async with` est ouvert -> modèle chargé 1× sur MPS, réutilisé.
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            app.state.session = session
            app.state.graph = build_rag_graph()
            worker = asyncio.create_task(_worker_loop(app))
            logger.info(f"event=ready device={agent_config.resolve_device(settings.device)} "
                        f"max_queue={settings.max_queue} rag_index={INDEX_DIR.exists()}")
            try:
                yield                      # <- l'API sert les requêtes ici
            finally:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                logger.info("event=shutdown")


app = FastAPI(title="FairPulse Agent API", version="0.4 (palier 1)", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Sonde de vivacité + diagnostic (device, modèles autorisés, index RAG présent)."""
    return {
        "status": "ok",
        "device": agent_config.resolve_device(settings.device),
        "allowed_models": ["PaPaGei-S", "Pulse-PPG"],
        "rag_index_present": INDEX_DIR.exists(),
    }


@app.post("/audit", status_code=202, response_model=JobCreated)
async def submit_audit(req: AuditRequest) -> JobCreated:
    """Crée un job d'audit et le met en file. Réponse immédiate (202) avec job_id.

    `req` est déjà validé par Pydantic (bornes, liste blanche de modèles) : une
    requête invalide n'arrive jamais ici (FastAPI renvoie 422 avant).
    """
    rec = app.state.store.create(req.model_dump())
    try:
        app.state.queue.put_nowait(rec.job_id)            # garde-fou : file bornée
    except asyncio.QueueFull as e:
        app.state.store.mark_error(rec.job_id, "file pleine")
        raise HTTPException(429, "File d'audit pleine — réessayez plus tard.") from e
    logger.info(f"job={rec.job_id} event=queued params={req.model_dump()}")
    return JobCreated(job_id=rec.job_id, status=rec.status)


@app.get("/audit/{job_id}", response_model=JobStatus)
def get_audit(job_id: str) -> JobStatus:
    """Statut + timings + (si terminé) rapport contextualisé."""
    rec = app.state.store.get(job_id)
    if rec is None:
        raise HTTPException(404, "job inconnu")
    return JobStatus(**rec.as_status())


@app.get("/audit/{job_id}/report.md")
def get_report_markdown(job_id: str) -> Response:
    """Renvoie le rapport markdown brut (pratique pour le lire/partager)."""
    rec = app.state.store.get(job_id)
    if rec is None:
        raise HTTPException(404, "job inconnu")
    if rec.status != "done":
        raise HTTPException(409, f"job non terminé (statut={rec.status})")
    return Response(rec.report_markdown or "", media_type="text/markdown")
