"""L'ÉTAT PARTAGÉ du graphe — le concept central de LangGraph.

Un StateGraph est une machine à états. L'« état » est un dictionnaire typé qui
CIRCULE de nœud en nœud. Chaque nœud reçoit l'état courant, et renvoie un petit
dictionnaire des champs qu'il veut METTRE À JOUR. LangGraph fusionne ce retour
dans l'état, puis passe au nœud suivant.

Règle de fusion (à bien comprendre) :
  - Par défaut, une clé renvoyée par un nœud ÉCRASE l'ancienne valeur.
  - Si on annote la clé avec un « reducer » (ci-dessous `log`), la valeur renvoyée
    est COMBINÉE avec l'ancienne au lieu de l'écraser. Ici on utilise `operator.add`
    sur une liste : chaque nœud renvoie `{"log": ["..."]}` et les messages
    s'ACCUMULENT au fil du graphe au lieu de se remplacer. C'est le mécanisme
    qu'utilisent les vrais agents pour accumuler un historique de messages.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AuditState(TypedDict, total=False):
    """État partagé qui transite par les 5 nœuds.

    `total=False` : aucun champ n'est obligatoire au départ. Le nœud d'entrée
    reçoit juste `config`, et chaque nœud ajoute progressivement sa contribution.
    """

    # --- Entrée (posée par run.py avant invoke) ---
    config: dict[str, Any]          # model_name, target, device, n_boot, seed, tailles d'échantillon

    # --- Rempli au fil des nœuds ---
    model: Any                      # nœud 1 : l'adapter PPG chargé (chargé UNE fois, sur MPS)
    segments: list[Any]             # nœud 2 : échantillon de fairpulse Segment (seedé)
    arrays: dict[str, Any]          # nœud 2 : labels / subjects / skin extraits en np.ndarray
    embeddings: Any                 # nœud 3 : matrice (N, 512) issue de l'inférence
    audit: dict[str, Any]           # nœud 4 : audit_embeddings(...) + losto(...)
    report_md: str                  # nœud 5 : rapport markdown rendu
    report_path: str                # nœud 5 : chemin du rapport écrit

    # --- Traçabilité : ACCUMULÉE grâce au reducer `operator.add` ---
    # Chaque nœud renvoie {"log": ["message"]} et la liste grandit au lieu d'être écrasée.
    log: Annotated[list[str], operator.add]
