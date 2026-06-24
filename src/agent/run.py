"""Point d'entrée CLI : construit l'état initial, exécute le graphe, montre le rapport.

    python -m agent.run --model PaPaGei-S --target spo2 \
        --max-patients-per-group 6 --n-boot 500 --seed 0

Ce que fait .invoke() : il part de l'état initial (juste `config`), exécute les
nœuds dans l'ordre des edges, fusionne chaque retour dans l'état, et renvoie
l'état FINAL (avec model/segments/embeddings/audit/report_md/log remplis).
"""

from __future__ import annotations

import argparse

from agent import config
from agent.graph import build_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent LangGraph — audit FairPulse (couche 1).")
    p.add_argument("--model", default="PaPaGei-S",
                   help="Modèle du zoo fairpulse (PaPaGei-S | Pulse-PPG).")
    p.add_argument("--target", default="spo2", choices=["spo2", "sao2"],
                   help="Cible de saturation à auditer.")
    p.add_argument("--device", default="auto",
                   help="auto (MPS si dispo) | mps | cpu.")
    p.add_argument("--max-patients-per-group", type=int, default=6,
                   help="Patients par teint (light/medium/dark) — borne le coût du run.")
    p.add_argument("--max-windows-per-encounter", type=int, default=8,
                   help="Fenêtres PPG max par encounter.")
    p.add_argument("--n-boot", type=int, default=500, help="Rééchantillonnages bootstrap.")
    p.add_argument("--seed", type=int, default=0, help="Graine (reproductibilité).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    # --- État INITIAL : on ne pose que `config`. Les nœuds rempliront le reste. ---
    initial_state = {
        "config": {
            "model_name": args.model,
            "target": args.target,
            "device": args.device,
            "max_patients_per_group": args.max_patients_per_group,
            "max_windows_per_encounter": args.max_windows_per_encounter,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "log": [],
    }

    app = build_graph()
    print(f"▶ Exécution du graphe d'audit ({args.model} / {args.target}) ...\n")

    # invoke() exécute START -> ... -> END et renvoie l'état final.
    final_state = app.invoke(initial_state)

    # Trace des nœuds (le reducer `operator.add` a accumulé tous les messages).
    print("\n".join(final_state["log"]))
    print("\n" + "=" * 72 + "\n")
    # Le rapport markdown produit par le dernier nœud.
    print(final_state["report_md"])
    print(f"\n✔ Rapport écrit : {final_state['report_path']}")


if __name__ == "__main__":
    main()
