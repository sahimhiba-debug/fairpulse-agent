"""fairpulse_agent — Couche 1 : agent LangGraph qui orchestre l'audit FairPulse.

Le package n'implémente AUCUNE logique d'audit : il importe et appelle le code
`fairpulse` existant (lecture seule). Son seul rôle est d'exprimer l'audit comme
un graphe d'étapes explicites (un StateGraph LangGraph) afin de pouvoir, aux
couches suivantes, y greffer MCP / RAG / une API sans réécrire le pipeline.
"""

# --- Bootstrap : rendre `import fairpulse` robuste (indépendant de l'état pip) ----
# On ajoute le `src/` du projet fairpulse au chemin d'import. Exécuté à l'import du
# package `agent`, donc AVANT tout sous-module qui ferait `import fairpulse`.
# FAIRPULSE_ROOT (env) permet de pointer un checkout FairPulse externe -> l'agent
# fonctionne aussi dans un dépôt DÉDIÉ (où `src/fairpulse` n'est pas embarqué).
import os as _os
import sys as _sys
from pathlib import Path as _Path

_root = _os.environ.get("FAIRPULSE_ROOT")
_FAIRPULSE_SRC = (_Path(_root) if _root else _Path(__file__).resolve().parents[3]) / "src"
if _FAIRPULSE_SRC.is_dir() and str(_FAIRPULSE_SRC) not in _sys.path:
    _sys.path.insert(0, str(_FAIRPULSE_SRC))
