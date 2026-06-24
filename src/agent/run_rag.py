"""Point d'entrée COUCHE 3 (RAG) : audit via MCP + contextualisation documentaire.

    PYTHONPATH=fairpulse_agent:fairpulse_agent/src PYTORCH_ENABLE_MPS_FALLBACK=1 \
        fairpulse_agent/.venv/bin/python -m agent.run_rag \
        --model PaPaGei-S --target spo2 --max-patients-per-group 6 --n-boot 500 --seed 0

Identique à run_mcp, mais avec le graphe RAG : un nœud retrieve_context interroge
le corpus et le rapport gagne une section « Contexte (littérature) » sourcée.
Prérequis : avoir construit l'index une fois (python -m rag.build_index).
"""

from __future__ import annotations

import argparse
import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agent import config
from agent.mcp_client import server_params
from agent.mcp_graph_rag import build_rag_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent LangGraph + MCP + RAG (couche 3).")
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

    app = build_rag_graph()
    print(f"▶ Couche 3 (MCP + RAG) — audit {args.model} / {args.target}\n")

    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
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
