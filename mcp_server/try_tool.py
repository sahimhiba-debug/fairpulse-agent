"""BONUS PÉDAGOGIQUE — tester le serveur MCP SANS l'agent LangGraph.

Montre que le serveur est une capacité autonome : on ouvre une session client,
on liste ce qu'il expose (tools / resources / prompts), et on appelle UN tool
isolément. C'est exactement ce que ferait n'importe quel client MCP.

    PYTORCH_ENABLE_MPS_FALLBACK=1 fairpulse_agent/.venv/bin/python \
        fairpulse_agent/mcp_server/try_tool.py

(Le serveur est lancé automatiquement en sous-processus via stdio.)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from agent.mcp_client import call_json, server_params  # noqa: E402


async def main() -> None:
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # poignée de main JSON-RPC

            # 1) Découverte : qu'expose le serveur ? (les 3 primitives)
            tools = await session.list_tools()
            resources = await session.list_resources()
            prompts = await session.list_prompts()
            print("TOOLS     :", [t.name for t in tools.tools])
            print("RESOURCES :", [str(r.uri) for r in resources.resources])
            print("PROMPTS   :", [p.name for p in prompts.prompts])

            # 2) Appel d'UN tool isolé : load_data_sample (pas besoin du modèle).
            print("\n→ call_tool load_data_sample(target='spo2', max_patients_per_group=6)")
            out = await call_json(session, "load_data_sample", {
                "target": "spo2", "max_patients_per_group": 6,
                "max_windows_per_encounter": 8, "seed": 0,
            })
            print("← réponse JSON du serveur :", out)

            # 3) Lecture d'une RESOURCE (donnée en lecture seule, par URI).
            res = await session.read_resource("fairpulse://zoo")
            print("\n→ read_resource fairpulse://zoo")
            print("← catalogue du zoo :\n" + res.contents[0].text)


if __name__ == "__main__":
    asyncio.run(main())
