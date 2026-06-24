"""PALIER 2 — persistance des jobs en base via SQLAlchemy (ORM).

Pourquoi une base ? Le store mémoire (palier 1) perd tout au redémarrage de l'API.
En prod, on veut que les jobs/résultats SURVIVENT (reprise après crash, plusieurs
process, audit/traçabilité). On utilise PostgreSQL via un ORM.

Concepts ORM (Object-Relational Mapping) :
• Une CLASSE Python (`JobRow`) est mappée à une TABLE SQL (`audit_jobs`) ; un objet
  = une ligne. On manipule des objets, l'ORM génère le SQL (INSERT/UPDATE/SELECT).
• `Session` = une unité de travail/transaction ; `commit()` valide.
• On ne met JAMAIS d'identifiant en dur : la connexion vient de DATABASE_URL (env).

`SqlJobStore` expose EXACTEMENT la même interface que le `JobStore` mémoire
(create/get/mark_running/add_timing/mark_done/mark_error) -> les endpoints et le
worker sont inchangés. On a juste swappé l'implémentation (principe d'inversion
de dépendance : le code dépend de l'interface, pas du backend).
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from api.jobs import JobRecord, _now_iso
from api.observability import logger


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    """Une ligne = un job d'audit. `JSON` stocke params/timings/result tels quels
    (JSONB côté Postgres). created_at/updated_at en ISO pour la traçabilité."""

    __tablename__ = "audit_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    params: Mapped[dict] = mapped_column(JSON)
    timings: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))


def _to_record(row: JobRow) -> JobRecord:
    """Convertit une ligne ORM en JobRecord (le type que les endpoints attendent)."""
    return JobRecord(
        job_id=row.job_id, status=row.status, params=row.params,
        timings=list(row.timings or []), error=row.error,
        report_markdown=row.report_markdown, result=row.result,
        created_at=row.created_at, updated_at=row.updated_at,
    )


class SqlJobStore:
    """JobStore persistant (PostgreSQL / SQLite) — interface identique au store mémoire."""

    def __init__(self, database_url: str):
        # SQLite (utilisé seulement pour des tests) a besoin de check_same_thread=False
        # car FastAPI accède à la base depuis plusieurs threads ; Postgres n'en a pas besoin.
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, future=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)            # crée la table si absente
        self._Session = sessionmaker(bind=self.engine, future=True)
        logger.info(f"event=db_ready backend={self.engine.dialect.name}")

    # NB : opérations synchrones (I/O DB local rapide). En forte charge on les
    # déporterait dans un threadpool (await asyncio.to_thread) pour ne pas bloquer
    # la boucle async — noté ici comme raffinement de prod.

    def create(self, params: dict) -> JobRecord:
        rec = JobRecord(job_id=uuid.uuid4().hex, status="queued", params=params)
        with self._Session() as s:
            s.add(JobRow(
                job_id=rec.job_id, status=rec.status, params=params, timings=[],
                created_at=rec.created_at, updated_at=rec.updated_at,
            ))
            s.commit()
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._Session() as s:
            row = s.get(JobRow, job_id)
            return _to_record(row) if row else None

    def _patch(self, job_id: str, **changes) -> None:
        with self._Session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                return
            for k, v in changes.items():
                setattr(row, k, v)
            row.updated_at = _now_iso()
            s.commit()

    def mark_running(self, job_id: str) -> None:
        self._patch(job_id, status="running")

    def add_timing(self, job_id: str, node: str, elapsed_ms: float) -> None:
        # On réassigne la liste entière (l'ORM ne détecte pas une mutation en place
        # d'une colonne JSON) -> garantit que l'UPDATE est bien émis.
        with self._Session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                return
            row.timings = list(row.timings or []) + [{"node": node, "elapsed_ms": round(elapsed_ms, 1)}]
            row.updated_at = _now_iso()
            s.commit()

    def mark_done(self, job_id: str, report_md: str, result: dict) -> None:
        self._patch(job_id, status="done", report_markdown=report_md, result=result)

    def mark_error(self, job_id: str, error: str) -> None:
        self._patch(job_id, status="error", error=error)
