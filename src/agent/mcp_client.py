"""Côté CLIENT MCP : comment se connecter au serveur et appeler un tool.

═══ CONCEPTS (côté client) ═════════════════════════════════════════════════════
• On décrit comment LANCER le serveur (StdioServerParameters : une commande +
  ses arguments). Le client démarre ce sous-processus et parle via stdin/stdout.
• stdio_client(params) ouvre le transport ; ClientSession est la conversation
  JSON-RPC. On fait `await session.initialize()` (poignée de main), puis
  `await session.call_tool(nom, arguments={...})`.
• call_tool renvoie un résultat dont le contenu textuel est du JSON : on le
  reparse en dict Python (helper call_json ci-dessous).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp import StdioServerParameters

# Chemin du script serveur (lancé par chemin de fichier ; il bootstrappe son path).
SERVER_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "server.py"


def server_params() -> StdioServerParameters:
    """Décrit COMMENT lancer le serveur MCP en sous-processus (transport stdio).

    On réutilise l'interpréteur courant (sys.executable = le venv dédié, donc torch
    + fairpulse + mcp sont présents). On propage le fallback MPS dans l'environnement
    du serveur (c'est LUI qui fait l'inférence GPU).
    """
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=env,
    )


async def call_json(session, name: str, arguments: dict) -> dict:
    """Appelle un tool et reparse sa sortie JSON en dict Python.

    Un tool FastMCP qui renvoie un dict le sérialise en JSON dans le 1er bloc de
    contenu textuel ; on le relit ici pour les nœuds de l'agent.
    """
    result = await session.call_tool(name, arguments=arguments)
    if result.isError:
        raise RuntimeError(f"Tool MCP '{name}' a échoué : {result.content}")
    return json.loads(result.content[0].text)
