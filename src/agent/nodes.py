"""Les 5 NŒUDS du graphe — un nœud = une étape de l'audit.

Anatomie d'un nœud LangGraph :
    def mon_noeud(state: AuditState) -> dict:
        ...  # lit ce dont il a besoin dans `state`
        return {"clef": valeur}   # ne renvoie QUE les champs à mettre à jour

LangGraph appelle ces fonctions dans l'ordre fixé par les edges (voir graph.py),
fusionne chaque retour dans l'état, et passe au suivant. Aucun nœud ne mute
`state` en place : il RENVOIE ses mises à jour (style fonctionnel = traçable,
rejouable). Tous les appels « métier » ci-dessous tapent dans le code fairpulse
existant — on orchestre, on ne réimplémente pas.
"""

from __future__ import annotations

# --- API fairpulse (lecture seule, importée telle quelle) ------------------------
from fairpulse.benchmark.evaluate import audit_embeddings, embed_segments
from fairpulse.benchmark.metrics import Estimate, subject_bootstrap
from fairpulse.benchmark.probe import fit_predict

from agent import config
from agent.report import render_markdown
from agent.sampling import SKIN_GROUPS, build_sample, per_group_counts
from agent.state import AuditState


# =================================================================================
# NŒUD 1 — charger le modèle PPG (une seule fois, sur MPS)
# =================================================================================
def load_model_node(state: AuditState) -> dict:
    """Instancie l'adapter fairpulse + charge les poids gelés sur le device choisi.

    Le modèle est chargé UNE fois ici puis transporté dans l'état : les nœuds
    suivants le réutilisent sans recharger (garde-fou « modèle chargé une fois »).
    """
    from fairpulse.zoo.adapters import load_model  # import local : torch via extra `deep`

    cfg = state["config"]
    device = config.resolve_device(cfg["device"])
    model = load_model(cfg["model_name"], weights_dir=config.WEIGHTS_DIR, device=device)

    # Garde-fou : sans poids, l'« audit » n'aurait aucun sens.
    if model.weights_path is None:
        raise FileNotFoundError(
            f"Aucun poids trouvé pour {cfg['model_name']} sous {config.WEIGHTS_DIR}."
        )

    return {
        "model": model,
        # On reflète le device RÉELLEMENT résolu dans la config (utile au rapport).
        "config": {**cfg, "device": device},
        "log": [f"[1/5] load_model : {model.name} chargé sur '{device}' "
                f"(poids: {model.weights_path})"],
    }


# =================================================================================
# NŒUD 2 — charger un petit échantillon de données de test (OpenOximetry, LOCAL)
# =================================================================================
def load_data_node(state: AuditState) -> dict:
    """Lit OpenOximetry en local, fenêtre le PPG, puis sous-échantillonne (seedé).

    ⚠️ Données restreintes : tout reste en local / in-process. On ne garde qu'un
    petit nombre de patients PAR groupe de teint (light/medium/dark) pour un run
    MPS rapide tout en conservant l'axe d'équité. Le sous-échantillonnage est
    déterministe (seed) -> reproductible.
    """
    cfg = state["config"]

    # Échantillonnage partagé (agent.sampling) — le MÊME code que le serveur MCP
    # de la couche 2, pour garantir des chiffres identiques entre les deux couches.
    sample, arrays = build_sample(
        target=cfg["target"],
        max_patients_per_group=cfg["max_patients_per_group"],
        max_windows_per_encounter=cfg["max_windows_per_encounter"],
        seed=cfg["seed"],
    )
    per_group = per_group_counts(arrays["skin"])
    n_pat = len(set(arrays["subjects"].tolist()))

    return {
        "segments": sample,
        "arrays": arrays,
        "log": [f"[2/5] load_data : {len(sample)} fenêtres / {n_pat} patients "
                f"(par teint: {per_group}) — échantillon seedé, local."],
    }


# =================================================================================
# NŒUD 3 — inférence : fenêtres PPG -> embeddings (sur MPS)
# =================================================================================
def run_inference_node(state: AuditState) -> dict:
    """Passe les fenêtres dans l'encodeur gelé -> matrice (N, 512).

    C'est l'unique étape « lourde » (GPU). embed_segments batche en interne et
    appelle model.embed(...), qui tourne sur le device fixé au nœud 1 (MPS).
    """
    model = state["model"]
    segments = state["segments"]

    embeddings = embed_segments(model, segments)  # (N, 512)

    return {
        "embeddings": embeddings,
        "log": [f"[3/5] run_inference : embeddings {embeddings.shape} "
                f"calculés sur '{state['config']['device']}'."],
    }


# =================================================================================
# NŒUD 4 — métriques + IC bootstrap + robustesse LOSTO
# =================================================================================
def _losto(embeddings, y, subjects, skin, n_boot, seed) -> dict[str, Estimate]:
    """Leave-One-Skin-Tone-Out (recopié de scripts/04 — patient-disjoint par nature).

    Test de ROBUSTESSE sous décalage de teint : on entraîne le probe Ridge sur
    2 groupes et on prédit le 3e (jamais vu). Si l'erreur explose sur le groupe
    tenu à l'écart, le modèle généralise mal hors de sa distribution de teint.
    """
    out: dict[str, Estimate] = {}
    for g in SKIN_GROUPS:
        test = skin == g
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        preds = fit_predict(embeddings[train], y[train], embeddings[test])
        out[g] = subject_bootstrap(y[test], preds, subjects[test], n_boot=n_boot, seed=seed)
    return out


def compute_metrics_node(state: AuditState) -> dict:
    """LOSO probe + suite de métriques (MAE + IC bootstrap sujet, calibration) + LOSTO.

    audit_embeddings fait déjà : leave-one-subject-out Ridge, MAE globale avec IC
    bootstrap AU NIVEAU SUJET (honnête : les fenêtres d'un patient sont corrélées),
    et fairness_skin = MAE par teint + IC + gap de disparité + calibration_bias
    (résidu signé moyen par groupe = sur/sous-estimation).

    Mapping demandé :
      - CALIBRATION  -> fairness_skin.calibration_bias (par teint)
      - ROBUSTESSE   -> LOSTO (généralisation hors teint vu)
      (by_activity reste vide ici : OpenOximetry n'a pas d'axe d'activité ;
       il se remplirait sur PPG-DaLiA.)
    """
    cfg = state["config"]
    emb = state["embeddings"]
    a = state["arrays"]

    audit = audit_embeddings(
        emb, a["labels"], a["subjects"],
        skin_groups=a["skin"],
        model=state["model"].name,
        dataset="OpenOximetry",
        task=f"{cfg['target']}_estimation",
        n_boot=cfg["n_boot"], seed=cfg["seed"],
    )
    # On ajoute la robustesse LOSTO au dict d'audit (clé propre à notre orchestration).
    audit["losto"] = _losto(emb, a["labels"], a["subjects"], a["skin"],
                            n_boot=cfg["n_boot"], seed=cfg["seed"])

    fr = audit["fairness_skin"]
    return {
        "audit": audit,
        "log": [f"[4/5] compute_metrics : MAE globale {audit['overall_mae']} ; "
                f"gap de disparité = {fr.gap:.2f} ; LOSTO sur {len(audit['losto'])} teints."],
    }


# =================================================================================
# NŒUD 5 — rédiger le rapport markdown (+ JSON) et l'écrire en local
# =================================================================================
def write_report_node(state: AuditState) -> dict:
    """Rend un rapport markdown structuré + un JSON machine, écrits dans reports/.

    Le rendu (agrégats uniquement — jamais de signaux bruts ni d'IDs patients)
    vit dans agent/report.py. write_json vient de fairpulse (réutilisé).
    """
    from fairpulse.audit.report import write_json

    cfg = state["config"]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"audit_{state['model'].name}_{cfg['target']}"

    md = render_markdown(state)
    md_path = config.REPORTS_DIR / f"{stem}.md"
    md_path.write_text(md + "\n")

    # JSON : on réutilise write_json de fairpulse (sérialise Estimate/FairnessReport).
    write_json(state["audit"], config.REPORTS_DIR / f"{stem}.json")

    return {
        "report_md": md,
        "report_path": str(md_path),
        "log": [f"[5/5] write_report : rapport écrit -> {md_path}"],
    }
