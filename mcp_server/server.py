"""Serveur MCP FairPulse — expose l'audit comme des primitives MCP.

═══ CONCEPTS MCP (à retenir) ═══════════════════════════════════════════════════
• Un SERVEUR MCP publie des capacités à des clients via JSON-RPC sur un TRANSPORT.
  Ici stdio : le client lance ce script en sous-processus et parle via stdin/stdout.
• FastMCP dérive AUTOMATIQUEMENT le schéma d'entrée d'un tool à partir des
  annotations de type des paramètres, et la description à partir de la docstring.
  Le schéma de sortie vient du type de retour (ici `dict` -> JSON structuré).
• Les 3 primitives MCP :
    - @mcp.tool()     ACTION invocable (peut calculer, avoir des effets). Nos audits.
    - @mcp.resource() DONNÉE en lecture seule, adressée par URI (style GET). Ex: le zoo.
    - @mcp.prompt()   GABARIT de workflow réutilisable que l'utilisateur déclenche.
• Une fois publiés, ces tools sont appelables par N'IMPORTE QUEL client MCP
  (Claude Desktop, Claude Code, notre agent LangGraph…), pas seulement notre agent.

⚠️ MPS : l'état lourd (modèle, embeddings) vit dans store.py, côté serveur. Le
modèle est chargé UNE fois sur MPS ; les clients n'échangent que des handles.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Bootstrap du sys.path : rendre `agent` et `mcp_server` importables ----------
# Le client nous lance par chemin de fichier (__package__ vaut None), donc on
# ajoute nous-mêmes les dossiers nécessaires avant les imports projet.
_AGENT_ROOT = Path(__file__).resolve().parents[1]          # .../fairpulse_agent
for _p in (_AGENT_ROOT, _AGENT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent import config  # noqa: E402, F401  (importé pour l'effet de bord : pose PYTORCH_ENABLE_MPS_FALLBACK)
from agent.sampling import SKIN_GROUPS, per_group_counts  # noqa: E402
from mcp_server import store  # noqa: E402

# Le serveur : un nom + (ici) réponses JSON structurées.
mcp = FastMCP("fairpulse-audit")

# Index RAG (couche 3) chargé paresseusement, UNE fois (le modèle d'embeddings
# côté store se charge sur MPS au 1er appel — comme le modèle PPG).
_RAG_STORE = None


def _rag_store():
    global _RAG_STORE
    if _RAG_STORE is None:
        from rag.store import VectorStore

        _RAG_STORE = VectorStore.load(_AGENT_ROOT / "rag" / "index")
    return _RAG_STORE


# --- Helper de sérialisation : Estimate (dataclass) -> dict JSON-able ------------
def _est(e) -> dict | None:
    """Un Estimate fairpulse (value + IC bootstrap + n) en dict sérialisable."""
    if e is None:
        return None
    return {"value": e.value, "ci_low": e.ci_low, "ci_high": e.ci_high,
            "n_subjects": e.n_subjects, "n_windows": e.n_windows}


# ═════════════════════════════════ TOOLS ════════════════════════════════════════

@mcp.tool()
def load_model(model_name: str = "PaPaGei-S", device: str = "auto") -> dict:
    """Charge un modèle PPG gelé du zoo fairpulse (sur MPS si dispo), une seule fois.

    Renvoie un *handle* opaque vers le modèle gardé côté serveur (l'objet torch ne
    transite jamais par le protocole), plus quelques métadonnées.
    """
    handle, _, _ = store.get_or_load_model(model_name, device)
    return store.model_meta(handle)


@mcp.tool()
def load_data_sample(target: str = "spo2", max_patients_per_group: int = 6,
                     max_windows_per_encounter: int = 8, seed: int = 0) -> dict:
    """Charge un petit échantillon OpenOximetry LOCAL (seedé), fenêtré pour l'audit.

    Données restreintes : restent côté serveur (local/in-process). Renvoie un handle
    + le décompte de fenêtres par teint (transparence sur ce qu'on audite).
    """
    handle, (_, arrays) = store.get_or_build_data(
        target, max_patients_per_group, max_windows_per_encounter, seed
    )
    subjects = arrays["subjects"]
    return {
        "data_handle": handle,
        "n_windows": int(len(subjects)),
        "n_patients": int(len(set(subjects.tolist()))),
        "per_group": per_group_counts(arrays["skin"]),
    }


@mcp.tool()
def run_inference(model_handle: str, data_handle: str) -> dict:
    """Passe les fenêtres PPG dans l'encodeur gelé -> embeddings (sur MPS), une fois.

    Étape « lourde » (GPU). Renvoie un handle vers la matrice gardée côté serveur.
    """
    handle, emb = store.get_or_infer(model_handle, data_handle)
    return {"emb_handle": handle, "shape": list(emb.shape)}


@mcp.tool()
def compute_metrics(emb_handle: str, data_handle: str, model_name: str = "PaPaGei-S",
                    target: str = "spo2", n_boot: int = 500, seed: int = 0) -> dict:
    """Performance GLOBALE : MAE [IC95 bootstrap sujet] + RMSE + corrélation de Pearson.

    Protocole leave-one-subject-out (probe Ridge sur embeddings gelés).
    """
    audit = store.get_or_audit(emb_handle, data_handle, model_name, target, n_boot, seed)
    return {
        "model": audit["model"], "dataset": audit["dataset"],
        "task": audit["task"], "protocol": audit["protocol"],
        "n_windows": audit["n_windows"], "n_subjects": audit["n_subjects"],
        "overall_mae": _est(audit["overall_mae"]),
        "rmse": audit["rmse"], "corr": audit["corr"],
    }


@mcp.tool()
def run_calibration_test(emb_handle: str, data_handle: str, model_name: str = "PaPaGei-S",
                         target: str = "spo2", n_boot: int = 500, seed: int = 0) -> dict:
    """ÉQUITÉ + CALIBRATION par teint : MAE [IC] par groupe, biais de calibration, gap.

    calibration_bias = résidu signé moyen (pred − vrai) par groupe : >0 sur-estime,
    <0 sous-estime. gap = disparité (groupe le plus dégradé − le meilleur).
    """
    audit = store.get_or_audit(emb_handle, data_handle, model_name, target, n_boot, seed)
    fr = audit["fairness_skin"]
    return {
        "group_key": fr.group_key,
        "per_group": {g: _est(e) for g, e in fr.per_group.items()},
        "calibration_bias": dict(fr.calibration_bias),
        "gap": fr.gap,
        "worst_group": fr.worst_group,
    }


@mcp.tool()
def run_robustness_test(emb_handle: str, data_handle: str,
                        n_boot: int = 500, seed: int = 0) -> dict:
    """ROBUSTESSE sous décalage de teint (leave-one-skin-tone-out).

    Pour chaque teint : on entraîne le probe sur les 2 autres et on prédit le teint
    tenu à l'écart (jamais vu). Une MAE qui explose => mauvaise généralisation hors
    distribution de teint.
    """
    from fairpulse.benchmark.metrics import subject_bootstrap
    from fairpulse.benchmark.probe import fit_predict

    emb = store.embeddings_for(emb_handle)
    arrays = store.arrays_for(data_handle)
    y, subjects, skin = arrays["labels"], arrays["subjects"], arrays["skin"]

    losto: dict[str, dict] = {}
    for g in SKIN_GROUPS:
        test = skin == g
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        preds = fit_predict(emb[train], y[train], emb[test])
        losto[g] = _est(subject_bootstrap(y[test], preds, subjects[test],
                                          n_boot=n_boot, seed=seed))
    return {"losto": losto}


@mcp.tool()
def retrieve_context(query: str, k: int = 4, min_score: float = 0.3) -> dict:
    """RAG : recherche sémantique dans le corpus documentaire FairPulse local.

    Renvoie les passages les plus proches (cosinus) AVEC leur source, pour
    contextualiser un résultat d'audit. ⚠️ Garde-fou anti-hallucination : seuls les
    passages dont le score dépasse `min_score` sont renvoyés ; si aucun, la liste est
    vide et `note` l'explique (l'agent dira « pas de contexte pertinent », sans rien
    inventer). On ne cite QUE ce qui est réellement dans le corpus.
    """
    hits = _rag_store().search(query, k=k, min_score=min_score)
    return {
        "query": query,
        "passages": [h.to_dict() for h in hits],
        "note": "" if hits else
                f"Aucun passage au-dessus du seuil min_score={min_score} : pas de contexte pertinent.",
    }


# ═══════════════════════════════ RESOURCE ═══════════════════════════════════════
# Une RESOURCE = donnée en lecture seule, adressée par URI. Pas d'effet de bord :
# un client la « lit » (comme un GET) pour du contexte. Ici, le catalogue du zoo.
@mcp.resource("fairpulse://zoo")
def zoo_catalog() -> str:
    """Liste les modèles PPG audibles (catalogue en lecture seule)."""
    import json

    from fairpulse.zoo import ZOO

    catalog = {
        name: {"embedding_dim": spec.embedding_dim,
               "input_fs_hz": spec.input_fs_hz,
               "input_seconds": spec.input_seconds}
        for name, spec in ZOO.items()
    }
    return json.dumps(catalog, indent=2)


@mcp.resource("rag://corpus")
def rag_corpus() -> str:
    """État du corpus RAG indexé (sources + nombre de passages) — lecture seule."""
    import json

    store = _rag_store()
    return json.dumps({"n_chunks": len(store.chunks), "sources": store.sources()},
                      ensure_ascii=False, indent=2)


# ════════════════════════════════ PROMPT ════════════════════════════════════════
# Un PROMPT = gabarit de workflow réutilisable, déclenché par l'utilisateur (ex.
# une commande slash dans Claude Desktop). Il GUIDE un client/LLM pour enchaîner
# les tools — il n'exécute rien lui-même.
@mcp.prompt()
def audit_model(model_name: str = "PaPaGei-S", target: str = "spo2") -> str:
    """Gabarit : auditer l'équité d'un modèle PPG via les tools de ce serveur."""
    return (
        f"Audite le modèle PPG « {model_name} » sur la cible « {target} » avec ce "
        f"serveur FairPulse, dans cet ordre :\n"
        f"1. load_model(model_name=\"{model_name}\")\n"
        f"2. load_data_sample(target=\"{target}\")\n"
        f"3. run_inference(model_handle, data_handle)\n"
        f"4. compute_metrics, run_calibration_test, run_robustness_test (mêmes handles)\n"
        f"Puis résume : MAE globale + IC, gap de disparité par teint, biais de "
        f"calibration, et robustesse LOSTO. Signale tout écart d'équité entre teints."
    )


if __name__ == "__main__":
    # Transport stdio par défaut : le client communique via stdin/stdout.
    mcp.run()
