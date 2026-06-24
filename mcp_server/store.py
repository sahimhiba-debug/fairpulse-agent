"""Registre STATEFUL du serveur MCP : handles -> objets lourds.

Pourquoi : un tool MCP échange du JSON. On ne peut pas faire transiter par JSON un
modèle torch ni une matrice d'embeddings. Donc le serveur garde ces objets EN
MÉMOIRE (dans ce module, vivant tout le temps que le serveur tourne) et ne renvoie
au client que des *handles* (petites chaînes). Le client réorchestre avec ces
handles ; l'état lourd ne quitte jamais le serveur.

Bonus : les handles sont DÉTERMINISTES (dérivés du contenu). Rappeler un tool avec
les mêmes arguments renvoie le même handle et réutilise le cache — le modèle MPS
n'est chargé qu'UNE fois, l'inférence ne tourne qu'UNE fois.
"""

from __future__ import annotations

import numpy as np

# config pose PYTORCH_ENABLE_MPS_FALLBACK avant tout import torch (cf. agent/config.py).
from agent import config
from agent.sampling import build_sample

# --- Les registres : de simples dicts, vivants pour la durée du serveur ----------
_MODELS: dict[str, object] = {}                         # handle -> adapter (sur MPS)
_DATASETS: dict[str, tuple[list, dict]] = {}            # handle -> (segments, arrays)
_EMBEDDINGS: dict[str, np.ndarray] = {}                 # handle -> (N, D)
_AUDITS: dict[str, dict] = {}                           # clé -> résultat audit_embeddings (cache)


# =================================================================================
# Modèle (chargé UNE fois, sur MPS)
# =================================================================================
def get_or_load_model(model_name: str, device: str = "auto"):
    """Charge l'adapter fairpulse + poids gelés, ou renvoie l'instance déjà en cache."""
    resolved = config.resolve_device(device)
    handle = f"model:{model_name}:{resolved}"
    if handle not in _MODELS:
        from fairpulse.zoo.adapters import load_model  # torch via extra `deep`

        model = load_model(model_name, weights_dir=config.WEIGHTS_DIR, device=resolved)
        if model.weights_path is None:
            raise FileNotFoundError(
                f"Aucun poids pour {model_name} sous {config.WEIGHTS_DIR}."
            )
        _MODELS[handle] = model
    return handle, _MODELS[handle], resolved


def model_meta(handle: str) -> dict:
    m = _MODELS[handle]
    return {"model_handle": handle, "name": m.name,
            "device": m.device, "weights_path": m.weights_path}


# =================================================================================
# Données (échantillon seedé, identique à la couche 1)
# =================================================================================
def get_or_build_data(target: str, max_patients_per_group: int,
                      max_windows_per_encounter: int, seed: int):
    handle = f"data:{target}:p{max_patients_per_group}:w{max_windows_per_encounter}:s{seed}"
    if handle not in _DATASETS:
        _DATASETS[handle] = build_sample(
            target, max_patients_per_group, max_windows_per_encounter, seed
        )
    return handle, _DATASETS[handle]


# =================================================================================
# Inférence (UNE fois, sur MPS) -> embeddings
# =================================================================================
def get_or_infer(model_handle: str, data_handle: str):
    handle = f"emb:({model_handle})|({data_handle})"
    if handle not in _EMBEDDINGS:
        from fairpulse.benchmark.evaluate import embed_segments

        model = _MODELS[model_handle]
        segments, _ = _DATASETS[data_handle]
        _EMBEDDINGS[handle] = embed_segments(model, segments)  # (N, D), sur MPS
    return handle, _EMBEDDINGS[handle]


# =================================================================================
# Audit complet (cache) — partagé par compute_metrics ET run_calibration_test
# =================================================================================
def get_or_audit(emb_handle: str, data_handle: str, model_name: str,
                 target: str, n_boot: int, seed: int) -> dict:
    """Lance audit_embeddings UNE fois (LOSO probe + suite de métriques) et met en cache.

    compute_metrics et run_calibration_test lisent ce même résultat -> chiffres
    cohérents et pas de re-calcul.
    """
    key = f"{emb_handle}|nb{n_boot}|s{seed}|{target}"
    if key not in _AUDITS:
        from fairpulse.benchmark.evaluate import audit_embeddings

        emb = _EMBEDDINGS[emb_handle]
        _, arrays = _DATASETS[data_handle]
        _AUDITS[key] = audit_embeddings(
            emb, arrays["labels"], arrays["subjects"],
            skin_groups=arrays["skin"],
            model=model_name, dataset="OpenOximetry",
            task=f"{target}_estimation", n_boot=n_boot, seed=seed,
        )
    return _AUDITS[key]


def arrays_for(data_handle: str) -> dict:
    return _DATASETS[data_handle][1]


def embeddings_for(emb_handle: str) -> np.ndarray:
    return _EMBEDDINGS[emb_handle]
