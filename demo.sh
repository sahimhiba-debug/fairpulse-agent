#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Démo FairPulse Agent — à enregistrer en GIF de terminal (asciinema + agg).
# Lance un audit de bout en bout (COUCHE 3 : LangGraph + MCP + RAG) puis affiche
# le début du rapport généré (équité par teint + contexte RAG cité).
#
# ⚠️ PRÉREQUIS LOCAUX (jamais committés) :
#   • FAIRPULSE_ROOT  = ton checkout FairPulse (code d'audit MIT)
#   • les POIDS PPG (PaPaGei-S) téléchargés localement
#   • les DONNÉES OpenOximetry (restreintes, DUA PhysioNet) stagées localement
#   Tout reste local / hors-ligne : rien n'est envoyé en ligne.
#
# Usage :  FAIRPULSE_ROOT=/chemin/vers/fairpulse ./demo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${FAIRPULSE_ROOT:?Définis FAIRPULSE_ROOT (export FAIRPULSE_ROOT=/chemin/vers/fairpulse)}"
PY="${PY:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.:src}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

step(){ printf "\n\033[1;36m▶ %s\033[0m\n" "$1"; sleep 1; }

step "FairPulse Agent — audit robustesse/équité PPG (couche 3 : LangGraph + MCP + RAG)"

step "1/2 · Audit de bout en bout (PaPaGei-S, cible spo2) — petit échantillon pour la démo"
"$PY" -m agent.run_rag --model PaPaGei-S --target spo2 \
      --max-patients-per-group 4 --n-boot 200 --seed 0

step "2/2 · Rapport généré — aperçu (équité par teint + contexte RAG cité)"
REPORT="$("$PY" - <<'PYEOF'
import glob, os
from agent import config
md = sorted(glob.glob(str(config.REPORTS_DIR / "audit_rag_*.md")), key=os.path.getmtime)
print(md[-1] if md else "")
PYEOF
)"
if [ -n "$REPORT" ]; then
  sed -n '1,42p' "$REPORT"
  printf "\n  … (rapport complet : %s)\n" "$REPORT"
else
  echo "  (aucun rapport trouvé sous reports/ — vérifie FAIRPULSE_ROOT / données)"
fi

step "Terminé ✓"
