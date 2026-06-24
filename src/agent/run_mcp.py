"""Point d'entrée COUCHE 2 : ouvre une session MCP, exécute le graphe via MCP.

    PYTHONPATH=fairpulse_agent/src PYTORCH_ENABLE_MPS_FALLBACK=1 \
        fairpulse_agent/.venv/bin/python -m agent.run_mcp \
        --model PaPaGei-S --target spo2 --max-patients-per-group 6 --n-boot 500 --seed 0

Déroulé :
  1. stdio_client(...) lance le serveur MCP en sous-processus.
  2. ClientSession + initialize() = la conversation MCP.
  3. On passe la session aux nœuds via le config LangGraph (configurable.session).
  4. app.ainvoke(...) exécute START→…→END ; chaque nœud appelle un tool MCP.
"""

from __future__ import annotations

import argparse
import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agent import config
from agent.mcp_client import server_params
from agent.mcp_graph import build_mcp_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent LangGraph via MCP — audit FairPulse (couche 2).")
    p.add_argument("--model", default="PaPaGei-S")
    p.add_argument("--target", default="spo2", choices=["spo2", "sao2"])
    p.add_argument("--device", default="auto")
    p.add_argument("--max-patients-per-group", type=int, default=6)
    p.add_argument("--max-windows-per-encounter", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    config.seed_everything(args.seed)

    initial_state = {
        "config": {
            "model_name": args.model, "target": args.target, "device": args.device,
            "max_patients_per_group": args.max_patients_per_group,
            "max_windows_per_encounter": args.max_windows_per_encounter,
            "n_boot": args.n_boot, "seed": args.seed,
        },
        "log": [],
    }

    app = build_mcp_graph()
    print(f"▶ Couche 2 (MCP) — audit {args.model} / {args.target}\n")

    # Ouverture du transport stdio + session : le serveur MCP est lancé ici.
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # On INJECTE la session dans le config : les nœuds la liront via
            # config["configurable"]["session"].
            run_config = {"configurable": {"session": session}}
            final_state = await app.ainvoke(initial_state, config=run_config)

    print("\n".join(final_state["log"]))
    print("\n" + "=" * 72 + "\n")
    print(final_state["report_md"])
    print(f"\n✔ Rapport écrit : {final_state['report_path']}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
