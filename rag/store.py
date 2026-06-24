"""INDEX VECTORIEL local + recherche par similarité cosinus.

Un index vectoriel stocke tous les vecteurs des chunks. Pour une requête, on
embarque sa question en vecteur, puis on cherche les chunks les plus PROCHES.

Similarité cosinus : mesure l'angle entre deux vecteurs (∈ [−1, 1] ; 1 = même
direction = même sens). Comme tous nos vecteurs sont NORMALISÉS (norme 1), le
cosinus se réduit à un simple PRODUIT SCALAIRE -> une multiplication matricielle
ultra rapide. À l'échelle de quelques centaines de chunks, numpy suffit largement
(pas besoin de FAISS, qui sert quand on a des millions de vecteurs).

Persistance : embeddings.npy (matrice) + chunks.jsonl (texte + métadonnées). On
construit l'index UNE fois (build_index.py), on le recharge ensuite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.embedder import embed_one


@dataclass
class Hit:
    text: str
    source: str
    section: str
    score: float  # similarité cosinus avec la requête (1 = identique)

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source,
                "section": self.section, "score": round(self.score, 4)}


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):
        self.embeddings = embeddings              # (N, D), normalisés
        self.chunks = chunks                      # liste de dicts {text, source, section, idx}

    # --- persistance -------------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with (index_dir / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        embeddings = np.load(index_dir / "embeddings.npy")
        chunks = [json.loads(line) for line in
                  (index_dir / "chunks.jsonl").read_text().splitlines() if line.strip()]
        return cls(embeddings, chunks)

    # --- recherche ---------------------------------------------------------------
    def search(self, query: str, k: int = 4, min_score: float = 0.3) -> list[Hit]:
        """Top-k chunks par cosinus, FILTRÉS par un seuil (garde-fou anti-hallucination).

        Si aucun chunk n'atteint ``min_score``, on renvoie une liste VIDE : l'agent
        dira « pas de contexte pertinent » plutôt que de citer un passage hors-sujet.
        """
        q = embed_one(query)                      # (D,) normalisé
        sims = self.embeddings @ q                # produit scalaire = cosinus (N,)
        order = np.argsort(-sims)[:k]             # les k plus grands scores
        hits: list[Hit] = []
        for i in order:
            score = float(sims[i])
            if score < min_score:
                break                             # trié -> les suivants sont pires
            c = self.chunks[int(i)]
            hits.append(Hit(c["text"], c["source"], c["section"], score))
        return hits

    def sources(self) -> dict[str, int]:
        """Nombre de chunks par fichier source (pour la resource rag://corpus)."""
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c["source"]] = out.get(c["source"], 0) + 1
        return out
