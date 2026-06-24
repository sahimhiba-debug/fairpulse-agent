"""Les nœuds de l'agent COUCHE 2 — ils appellent les tools VIA MCP.

Même topologie de graphe qu'en couche 1, mais au lieu d'importer les fonctions
fairpulse, chaque nœud fait `await session.call_tool(...)`. La session MCP est
fournie par LangGraph via le `config` du run (config["configurable"]["session"]).

Signature LangGraph d'un nœud avec accès au config :
    async def mon_noeud(state, config) -> dict
LangGraph injecte le RunnableConfig en 2e argument ; on y lit la session ouverte
dans run_mcp.py. Les nœuds sont ASYNC car le protocole MCP est asynchrone.
"""

from __future__ import annotations

from fairpulse.benchmark.fairness import FairnessReport
from fairpulse.benchmark.metrics import Estimate

from agent.mcp_client import call_json
from agent.report import render_markdown


def _session(config):
    """Récupère la session MCP injectée par LangGraph dans le config du run."""
    return config["configurable"]["session"]


# ───────────────────────── Nœud 1 : tool load_model ─────────────────────────────
async def load_model_node(state, config) -> dict:
    cfg = state["config"]
    info = await call_json(_session(config), "load_model", {
        "model_name": cfg["model_name"], "device": cfg["device"],
    })
    return {
        "model_info": info,
        "log": [f"[1/7] load_model (MCP) : {info['name']} sur '{info['device']}' "
                f"-> handle={info['model_handle']}"],
    }


# ──────────────────────── Nœud 2 : tool load_data_sample ────────────────────────
async def load_data_node(state, config) -> dict:
    cfg = state["config"]
    info = await call_json(_session(config), "load_data_sample", {
        "target": cfg["target"],
        "max_patients_per_group": cfg["max_patients_per_group"],
        "max_windows_per_encounter": cfg["max_windows_per_encounter"],
        "seed": cfg["seed"],
    })
    return {
        "data_info": info,
        "log": [f"[2/7] load_data_sample (MCP) : {info['n_windows']} fenêtres / "
                f"{info['n_patients']} patients {info['per_group']} "
                f"-> handle={info['data_handle']}"],
    }


# ───────────────────────── Nœud 3 : tool run_inference ──────────────────────────
async def run_inference_node(state, config) -> dict:
    info = await call_json(_session(config), "run_inference", {
        "model_handle": state["model_info"]["model_handle"],
        "data_handle": state["data_info"]["data_handle"],
    })
    return {
        "emb_info": info,
        "log": [f"[3/7] run_inference (MCP) : embeddings {info['shape']} (MPS, côté serveur) "
                f"-> handle={info['emb_handle']}"],
    }


# ──────────────────────── Nœud 4 : tool compute_metrics ─────────────────────────
async def compute_metrics_node(state, config) -> dict:
    cfg = state["config"]
    metrics = await call_json(_session(config), "compute_metrics", {
        "emb_handle": state["emb_info"]["emb_handle"],
        "data_handle": state["data_info"]["data_handle"],
        "model_name": cfg["model_name"], "target": cfg["target"],
        "n_boot": cfg["n_boot"], "seed": cfg["seed"],
    })
    om = metrics["overall_mae"]
    return {
        "metrics": metrics,
        "log": [f"[4/7] compute_metrics (MCP) : MAE globale "
                f"{om['value']:.2f} [{om['ci_low']:.2f}, {om['ci_high']:.2f}]"],
    }


# ────────────────────── Nœud 5 : tool run_calibration_test ──────────────────────
async def calibration_node(state, config) -> dict:
    cfg = state["config"]
    calib = await call_json(_session(config), "run_calibration_test", {
        "emb_handle": state["emb_info"]["emb_handle"],
        "data_handle": state["data_info"]["data_handle"],
        "model_name": cfg["model_name"], "target": cfg["target"],
        "n_boot": cfg["n_boot"], "seed": cfg["seed"],
    })
    return {
        "calibration": calib,
        "log": [f"[5/7] run_calibration_test (MCP) : gap={calib['gap']:.2f} "
                f"worst={calib['worst_group']}"],
    }


# ────────────────────── Nœud 6 : tool run_robustness_test ───────────────────────
async def robustness_node(state, config) -> dict:
    cfg = state["config"]
    robust = await call_json(_session(config), "run_robustness_test", {
        "emb_handle": state["emb_info"]["emb_handle"],
        "data_handle": state["data_info"]["data_handle"],
        "n_boot": cfg["n_boot"], "seed": cfg["seed"],
    })
    return {
        "robustness": robust,
        "log": [f"[6/7] run_robustness_test (MCP) : LOSTO sur {len(robust['losto'])} teints"],
    }


# ─────────── Nœud RAG (couche 3) : tool retrieve_context, requêtes ciblées ───────
async def retrieve_context_node(state, config) -> dict:
    """Forme des requêtes À PARTIR DES RÉSULTATS, puis récupère le contexte (RAG).

    On ne récupère pas « au hasard » : chaque requête est dérivée d'une observation
    chiffrée de l'audit (groupe le plus dégradé, sens de la calibration, équité des
    foundation models). Les passages récupérés (réels, sourcés) serviront à
    contextualiser le rapport. Anti-hallucination : si le tool ne renvoie rien
    au-dessus du seuil, on n'invente pas.
    """
    session = _session(config)
    calib = state["calibration"]
    worst = calib.get("worst_group", "dark")
    sign = "sous-estimation" if calib.get("calibration_bias", {}).get(worst, 0) < 0 \
        else "surestimation"

    # 2-3 requêtes ciblées sur ce qu'on vient de mesurer.
    queries = [
        f"biais d'oxymétrie SpO2 SaO2 {sign} peau foncée hypoxémie occulte",
        f"équité des foundation models PPG erreur plus élevée groupe {worst} couleur de peau",
        "leave-one-skin-tone-out robustesse audit équité teinte de peau",
    ]

    seen: set[tuple[str, str]] = set()
    passages: list[dict] = []
    for q in queries:
        out = await call_json(session, "retrieve_context",
                              {"query": q, "k": 3, "min_score": 0.35})
        for p in out["passages"]:
            key = (p["source"], p["section"])
            if key not in seen:                  # dédup par (fichier, section)
                seen.add(key)
                passages.append({**p, "matched_query": q})

    passages.sort(key=lambda p: -p["score"])     # meilleurs scores d'abord
    return {
        "context": {"queries": queries, "passages": passages},
        "log": [f"[RAG] retrieve_context : {len(passages)} passages pertinents "
                f"(sur {len(queries)} requêtes dérivées des résultats)"],
    }


# ───────────────── Nœud 7 : rapport (LOCAL — pas de tool MCP) ────────────────────
def _est(d: dict | None) -> Estimate | None:
    """Reconstruit un Estimate fairpulse à partir du JSON renvoyé par un tool."""
    return Estimate(**d) if d else None


def _rebuild_audit(state) -> dict:
    """Réassemble le dict `audit` au MÊME format que la couche 1, à partir des JSON.

    But : pouvoir RÉUTILISER tel quel le render_markdown de la couche 1 -> rapport
    identique. On reconstruit les dataclasses Estimate / FairnessReport.
    """
    m, calib, robust = state["metrics"], state["calibration"], state["robustness"]
    return {
        "model": m["model"], "dataset": m["dataset"], "task": m["task"],
        "protocol": m["protocol"], "n_windows": m["n_windows"], "n_subjects": m["n_subjects"],
        "overall_mae": _est(m["overall_mae"]), "rmse": m["rmse"], "corr": m["corr"],
        "by_activity": {},
        "fairness_skin": FairnessReport(
            group_key=calib["group_key"],
            per_group={g: _est(e) for g, e in calib["per_group"].items()},
            gap=calib["gap"],
            calibration_bias=calib["calibration_bias"],
            worst_group=calib["worst_group"],
        ),
        "losto": {g: _est(e) for g, e in robust["losto"].items()},
    }


def write_report_node(state) -> dict:
    """Rend le rapport (réutilise render_markdown couche 1) + JSON, en local."""
    from fairpulse.audit.report import write_json

    from agent import config

    # device RÉSOLU (mps) renvoyé par le serveur, pour un rapport identique à la couche 1.
    cfg = {**state["config"], "device": state["model_info"]["device"]}
    audit = _rebuild_audit(state)

    # render_markdown lit state["config"], state["audit"], state.get("context"), state.get("log").
    render_state = {"config": cfg, "audit": audit,
                    "context": state.get("context"), "log": state["log"]}
    md = render_markdown(render_state)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # nom distinct selon la couche : "rag" si contexte documentaire, sinon "mcp".
    layer = "rag" if state.get("context") else "mcp"
    stem = f"audit_{layer}_{audit['model']}_{cfg['target']}"
    md_path = config.REPORTS_DIR / f"{stem}.md"
    md_path.write_text(md + "\n")
    write_json(audit, config.REPORTS_DIR / f"{stem}.json")

    return {
        "report_md": md,
        "report_path": str(md_path),
        "log": [f"[7/7] write_report (local) : rapport écrit -> {md_path}"],
    }
