"""Modèle d'EMBEDDINGS de texte — sentence-transformers sur MPS, chargé UNE fois.

Un embedding de texte = un vecteur (ici 384 nombres) qui capture le SENS d'un
passage. Deux textes au sens proche -> vecteurs proches. C'est ce qui permet la
recherche « sémantique » (par le sens) plutôt que par mots-clés exacts.

⚠️ Calculer un embedding = faire de l'inférence : on charge le modèle UNE fois,
sur MPS si dispo (comme le modèle PPG des couches 1-2).

Convention importante : on NORMALISE les vecteurs (norme 1). Du coup la similarité
cosinus se calcule par un simple produit scalaire (cf. store.py).
"""

from __future__ import annotations

import numpy as np

from agent import config  # pose PYTORCH_ENABLE_MPS_FALLBACK + resolve_device + seeds

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingue (corpus en FR), 384-dim

_MODEL = None  # singleton : le modèle n'est instancié qu'une fois par processus


def _load():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        config.seed_everything(0)
        device = config.resolve_device("auto")  # "mps" si dispo
        _MODEL = SentenceTransformer(MODEL_NAME, device=device)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode une liste de textes -> matrice (N, 384) de vecteurs NORMALISÉS."""
    model = _load()
    # normalize_embeddings=True -> norme 1 -> cosinus = produit scalaire ensuite.
    embs = model.encode(texts, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    """Encode une seule requête -> vecteur (384,) normalisé."""
    return embed_texts([text])[0]


def device() -> str:
    return _load().device.type if _MODEL is not None else config.resolve_device("auto")
