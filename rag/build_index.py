"""Construction de l'index RAG (one-shot) : chunk -> embed (MPS) -> sauvegarde.

    PYTORCH_ENABLE_MPS_FALLBACK=1 fairpulse_agent/.venv/bin/python \
        -m rag.build_index

⚠️ Le PREMIER appel télécharge le modèle d'embeddings (open, ~470 Mo, HuggingFace),
puis il est mis en cache. Le CORPUS, lui, ne télécharge rien : ce sont tes docs
locaux + ce que tu déposes dans rag/corpus/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap du path (rendre `agent` et `rag` importables si lancé par chemin).
_AGENT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_AGENT_ROOT, _AGENT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent import config  # noqa: E402
from rag.chunker import chunk_corpus  # noqa: E402
from rag.embedder import embed_texts  # noqa: E402
from rag.store import VectorStore  # noqa: E402

INDEX_DIR = _AGENT_ROOT / "rag" / "index"
CORPUS_DIR = _AGENT_ROOT / "rag" / "corpus"


def corpus_paths() -> list[Path]:
    """Rassemble le corpus : docs FairPulse locaux + dossier corpus/ extensible.

    Aucun téléchargement : uniquement des fichiers DÉJÀ présents en local.
    """
    paths: list[Path] = []
    # 1) Tes documents FairPulse (la base documentaire de référence).
    docs = config.FAIRPULSE_ROOT / "docs"
    paths += sorted(docs.glob("*.md"))
    readme = config.FAIRPULSE_ROOT / "README.md"
    if readme.exists():
        paths.append(readme)
    # 2) Tes sources additionnelles (drop-folder) : .md / .txt que TU ajoutes.
    #    (on saute le README d'instructions du dossier, qui n'est pas une source).
    if CORPUS_DIR.exists():
        extra = sorted(CORPUS_DIR.glob("*.md")) + sorted(CORPUS_DIR.glob("*.txt"))
        paths += [p for p in extra if p.name.lower() != "readme.md"]
    return paths


def main() -> None:
    paths = corpus_paths()
    print(f"Corpus : {len(paths)} fichiers")
    for p in paths:
        print(f"  - {p.relative_to(config.FAIRPULSE_ROOT)}")

    chunks = chunk_corpus(paths)
    print(f"\nChunking -> {len(chunks)} passages (~150 mots, overlap 1 phrase)")

    print(f"Embedding sur '{config.resolve_device('auto')}' (modèle chargé 1×) ...")
    embeddings = embed_texts([c.text for c in chunks])
    print(f"Embeddings : {embeddings.shape} (normalisés)")

    store = VectorStore(embeddings, [c.to_dict() for c in chunks])
    store.save(INDEX_DIR)
    print(f"\n✔ Index écrit -> {INDEX_DIR}")
    print("  Par source :", store.sources())


if __name__ == "__main__":
    main()
