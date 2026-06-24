"""Découpage des documents en CHUNKS (passages).

Pourquoi chunker ? On veut récupérer *une idée précise et citable*, pas un document
entier. Compromis sur la taille :
  • trop GROS  -> le passage noie l'info pertinente, l'embedding devient « moyen » et flou ;
  • trop PETIT -> on perd le contexte (une phrase isolée n'est plus interprétable).
Cible : ~150 mots par chunk, avec un léger CHEVAUCHEMENT (la dernière phrase du chunk
précédent est répétée au début du suivant) pour ne pas couper une idée à la frontière.

On garde des MÉTADONNÉES par chunk (fichier source + titre de section) : c'est ce qui
permettra de CITER précisément la source dans le rapport.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TARGET_WORDS = 150      # taille visée d'un chunk (en mots)
OVERLAP_SENTENCES = 1   # nb de phrases reprises du chunk précédent (chevauchement)


@dataclass
class Chunk:
    text: str
    source: str     # nom du fichier (pour la citation)
    section: str     # titre de section markdown le plus proche
    idx: int         # index du chunk dans le corpus

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source,
                "section": self.section, "idx": self.idx}


def _split_sentences(text: str) -> list[str]:
    """Découpe naïve en phrases (suffisant pour le chevauchement)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _flush(buffer: list[str], source: str, section: str, idx: int) -> Chunk | None:
    if not buffer:
        return None
    return Chunk(" ".join(buffer).strip(), source, section or "(préambule)", idx)


def chunk_markdown(path: Path) -> list[Chunk]:
    """Découpe un .md en passages, en suivant les titres de section (#, ##, ###).

    Stratégie : on parcourt ligne à ligne ; un titre met à jour la « section
    courante » (mémorisée pour la citation). On accumule le texte par paragraphe
    jusqu'à atteindre ~TARGET_WORDS, puis on émet un chunk et on repart avec un
    chevauchement d'une phrase.
    """
    source = path.name
    chunks: list[Chunk] = []
    section = ""
    buffer: list[str] = []
    words = 0

    def emit():
        nonlocal buffer, words
        c = _flush(buffer, source, section, len(chunks))
        if c is not None:
            chunks.append(c)
        # chevauchement : on garde la dernière phrase pour le prochain chunk
        tail = _split_sentences(" ".join(buffer))[-OVERLAP_SENTENCES:] if buffer else []
        buffer = list(tail)
        words = sum(len(s.split()) for s in buffer)

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        header = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header:
            emit()                       # un nouveau titre clôt le chunk en cours
            section = header.group(2).strip()
            continue
        if not line.strip():             # ligne vide = frontière de paragraphe
            if words >= TARGET_WORDS:
                emit()
            continue
        buffer.append(line.strip())
        words += len(line.split())
        if words >= TARGET_WORDS:
            emit()

    emit()  # dernier chunk
    # Filtre les chunks trop courts (titres orphelins, lignes de tableau isolées)
    return [c for c in chunks if len(c.text.split()) >= 12]


def chunk_corpus(paths: list[Path]) -> list[Chunk]:
    """Chunk tous les fichiers et réindexe proprement (idx global)."""
    out: list[Chunk] = []
    for p in paths:
        for c in chunk_markdown(p):
            out.append(Chunk(c.text, c.source, c.section, len(out)))
    return out
