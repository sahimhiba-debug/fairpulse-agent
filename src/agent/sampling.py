"""Échantillonnage OpenOximetry — DÉTERMINISTE et PARTAGÉ.

Extrait ici pour être utilisé À LA FOIS par la couche 1 (appel direct) et la
couche 2 (serveur MCP). Même code + même seed => même échantillon => rapports
numériquement identiques entre les deux couches. C'est ce qui rend la migration
vers MCP vérifiable : on prouve que rien n'a changé.

⚠️ Données restreintes : tout reste local / in-process.
"""

from __future__ import annotations

import numpy as np

from fairpulse.data.openoximetry import OpenOximetry

from agent import config

SKIN_GROUPS = ("light", "medium", "dark")  # ordre Monk (clair -> foncé)


def build_sample(target: str, max_patients_per_group: int,
                 max_windows_per_encounter: int, seed: int):
    """Construit un petit échantillon seedé : renvoie (segments, arrays).

    arrays = {"labels": (N,), "subjects": (N,), "skin": (N,)} (np.ndarray).
    """
    ox = OpenOximetry(config.OPENOX_ROOT)
    if not ox.is_available():
        raise FileNotFoundError(
            f"OpenOximetry non stagé sous {config.OPENOX_ROOT} (données restreintes)."
        )

    segments = ox.build_segments(
        target=target, max_windows_per_encounter=max_windows_per_encounter
    )
    if not segments:
        raise RuntimeError("Aucun segment exploitable (PPG + teint + saturation).")

    # --- Sous-échantillonnage déterministe : N patients par groupe de teint ------
    rng = np.random.default_rng(seed)
    patients_by_group: dict[str, list[str]] = {g: [] for g in SKIN_GROUPS}
    for seg in segments:
        g = seg.skin_group
        if g in patients_by_group and seg.subject_id not in patients_by_group[g]:
            patients_by_group[g].append(seg.subject_id)

    keep_patients: set[str] = set()
    for g in SKIN_GROUPS:
        pool = sorted(patients_by_group[g])          # tri = ordre stable avant tirage
        k = min(max_patients_per_group, len(pool))
        chosen = rng.choice(pool, size=k, replace=False) if k else []
        keep_patients.update(map(str, chosen))

    sample = [s for s in segments if s.subject_id in keep_patients]

    arrays = {
        "labels": np.array([s.label for s in sample], dtype=float),
        "subjects": np.array([s.subject_id for s in sample], dtype=object),
        "skin": np.array([s.skin_group for s in sample], dtype=object),
    }
    return sample, arrays


def per_group_counts(skin: np.ndarray) -> dict[str, int]:
    """Nombre de fenêtres par groupe de teint (pour les logs / rapports)."""
    return {g: int(np.sum(skin == g)) for g in SKIN_GROUPS}
