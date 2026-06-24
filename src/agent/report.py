"""Rendu du rapport d'audit en markdown.

Séparé des nœuds pour garder `nodes.py` centré sur le flux LangGraph. Le rapport
ne contient QUE des agrégats (MAE, IC, gap, calibration, taux) — jamais de signal
PPG brut ni d'identifiant patient (garde-fou données restreintes).
"""

from __future__ import annotations

from agent.state import AuditState

UNIT = {"spo2": "%SpO2", "sao2": "%SaO2"}  # l'erreur d'une saturation est en points de %


def _context_section(context: dict | None) -> list[str]:
    """Rend la section « Contexte (littérature) » à partir des passages RAG récupérés.

    ⚠️ Anti-hallucination : on ne cite QUE les passages réellement renvoyés par le
    retrieval (texte + source + score). Si la liste est vide, on l'écrit franchement.
    """
    if context is None:
        return []  # couches 1-2 : pas de RAG, pas de section.

    out = ["", "## Contexte (littérature) — RAG",
           "_Passages récupérés par recherche sémantique dans le corpus FairPulse local, "
           "pour contextualiser les chiffres ci-dessus. Citations = sources réelles du corpus._", ""]

    passages = context.get("passages", [])
    if not passages:
        out += ["> _Aucun passage documentaire pertinent trouvé (score sous le seuil). "
                "Pas de contexte ajouté — aucune référence inventée._"]
        return out

    for p in passages:
        excerpt = " ".join(p["text"].split())            # normalise les espaces
        if len(excerpt) > 320:
            excerpt = excerpt[:320].rsplit(" ", 1)[0] + " …"
        out += [
            f"- **{p['source']} § {p['section']}**  _(similarité {p['score']:.2f})_",
            f"  > {excerpt}",
        ]
    out += ["",
            "_Source de chaque passage = fichier du corpus local ci-dessus. Les références "
            "bibliographiques complètes (Sjoding 2020, Fawzy 2022, …) figurent dans "
            "`rapport_v1.md` §5.6._"]
    return out


def render_markdown(state: AuditState) -> str:
    cfg = state["config"]
    audit = state["audit"]
    fr = audit["fairness_skin"]
    losto = audit.get("losto", {})
    unit = UNIT.get(cfg["target"], "")

    lines = [
        f"# Rapport d'audit FairPulse — {audit['model']}",
        "",
        f"- **Modèle audité** : {audit['model']}  ·  **device** : `{cfg['device']}`",
        f"- **Données** : {audit['dataset']} (local, restreint — audit hors-ligne)  "
        f"·  **tâche** : {audit['task']}",
        f"- **Protocole** : {audit['protocol']}  ·  **n_boot** : {cfg['n_boot']}  "
        f"·  **seed** : {cfg['seed']}",
        f"- **Couverture** : **{audit['n_windows']}** fenêtres / "
        f"**{audit['n_subjects']}** patients",
        "",
        "## Performance globale",
        f"- **MAE** ({unit}), IC95 bootstrap (niveau patient) : **{audit['overall_mae']}**",
        f"- RMSE : {audit['rmse']:.2f}  ·  Pearson r : {audit['corr']:.3f}",
        "",
        "## Équité par teint de peau (Monk) + calibration",
        f"_Gap de disparité = **{fr.gap:.2f} {unit}** · groupe le plus dégradé : "
        f"**{fr.worst_group}**_",
        "",
        f"| Teint | MAE [IC95] | Biais calibration ({unit}) | LOSTO MAE [IC95] |",
        "|---|---|---|---|",
    ]
    # Une ligne par groupe : MAE intra (per_group) + calibration + robustesse LOSTO.
    for g in ("light", "medium", "dark"):
        if g not in fr.per_group:
            continue
        calib = fr.calibration_bias.get(g, float("nan"))
        losto_cell = str(losto[g]) if g in losto else "n/a"
        lines.append(f"| {g} | {fr.per_group[g]} | {calib:+.2f} | {losto_cell} |")

    lines += [
        "",
        "### Lecture",
        "- **MAE [IC95]** : erreur intra-groupe (leave-one-subject-out), IC ré-échantillonné "
        "au niveau patient.",
        "- **Biais calibration** : résidu signé moyen (pred − vrai). `>0` = sur-estime la "
        "saturation, `<0` = sous-estime.",
        "- **LOSTO** : *leave-one-skin-tone-out* — erreur sur un teint **jamais vu** à "
        "l'entraînement (robustesse sous décalage de teint).",
        "",
        "> ⚠️ OpenOximetry est une donnée restreinte (DUA). Cet audit tourne strictement "
        "en local / in-process ; ce rapport ne contient que des métriques agrégées. "
        "Label au niveau *encounter* (le PPG n'est pas time-synced à SpO2/SaO2 en v1.1.1) : "
        "valeurs absolues à interpréter comme un ordre de grandeur, pas une vérité par fenêtre.",
    ]

    # --- Section RAG (couche 3) : insérée seulement si un contexte a été récupéré.
    # Rétro-compatible : couches 1-2 n'ont pas de "context" -> section absente.
    lines += _context_section(state.get("context"))

    lines += [
        "",
        "---",
        "### Trace d'exécution du graphe",
        "```",
        *state.get("log", []),
        "```",
    ]
    return "\n".join(lines)
