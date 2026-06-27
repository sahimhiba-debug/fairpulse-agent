#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Démo VITRINE (publique) — FairPulse Agent.  À enregistrer en GIF pour le README.
#
# ⚠️ POURQUOI CE SCRIPT EXISTE :
#   demo.sh (l'autre) lance un VRAI audit et affiche un rapport calculé sur les
#   données OpenOximetry (RESTREINTES, DUA PhysioNet) -> il NE DOIT JAMAIS être
#   filmé en public.  CE script-ci est 100 % autonome : il NARRE le flux du
#   pipeline (libellés uniquement), n'exécute AUCUN audit, ne lit AUCUNE donnée
#   réelle, ne fait AUCUN appel réseau, et ne dépend NI de FAIRPULSE_ROOT, NI des
#   poids, NI des données.  Le seul aperçu chiffré provient du rapport
#   SYNTHÉTIQUE déjà étiqueté (docs/examples/rapport_exemple.md).
#
# Usage :  ./demo_public.sh        (depuis la racine du dépôt fairpulse-agent)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Couleurs (désactivées si la sortie n'est pas un terminal).
if [ -t 1 ]; then
  C="\033[1;36m"; G="\033[1;32m"; Y="\033[1;33m"; D="\033[2m"; R="\033[0m"
else
  C=""; G=""; Y=""; D=""; R=""
fi

# Localise le rapport synthétique par rapport à ce script (robuste au cwd).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNTH="$HERE/docs/examples/rapport_exemple.md"

PAUSE="${PAUSE:-1.4}"   # pause entre étapes (surchargeable : PAUSE=0.8 ./demo_public.sh)

step(){ printf "${C}▶ %s${R}\n" "$1"; sleep "$PAUSE"; }
note(){ printf "${D}   %s${R}\n" "$1"; }

clear 2>/dev/null || true
printf "${G}%s${R}\n" "════════════════════════════════════════════════════════════════"
printf "${G} FairPulse Agent — audit robustesse & équité PPG${R}\n"
printf "${G} Couche 3 : LangGraph + MCP + RAG${R}\n"
printf "${G}%s${R}\n" "════════════════════════════════════════════════════════════════"
printf "${Y} ⚠️  DÉMO ILLUSTRATIVE — narration du flux, aucune donnée réelle.${R}\n"
sleep "$PAUSE"

# ── Narration des 8 nœuds du graphe (mêmes étapes que build_rag_graph) ──────────
step "[1/8] load_model — chargement du modèle PPG via MCP"
note  "PaPaGei-S · device=mps (Apple Silicon)"

step "[2/8] load_data — échantillon (démo, illustratif)"
note  "teints Monk : light / medium / dark"

step "[3/8] run_inference — embeddings PPG (MPS, côté serveur MCP)"
note  "fenêtres -> vecteurs d'embedding"

step "[4/8] compute_metrics — MAE + IC95 bootstrap (niveau patient)"
note  "IC ré-échantillonné par patient (pas par fenêtre)"

step "[5/8] run_calibration_test — biais & équité par teint de peau"
note  "gap de disparité + biais signé par groupe"

step "[6/8] run_robustness_test — leave-one-skin-tone-out (LOSTO)"
note  "erreur sur un teint jamais vu à l'entraînement"

step "[7/8] retrieve_context — RAG : contexte cité (extractif, SANS LLM)"
note  "MiniLM 384-d · cosinus · top-k filtré par seuil (anti-hallucination)"

step "[8/8] write_report — rapport markdown + JSON"
note  "agrégats uniquement (jamais de signal brut)"

# ── Aperçu du rapport SYNTHÉTIQUE (étiqueté) ───────────────────────────────────
printf "\n${G}── Aperçu du rapport (EXEMPLE SYNTHÉTIQUE) ──────────────────────${R}\n\n"
sleep "$PAUSE"
if [ -f "$SYNTH" ]; then
  sed -n '1,34p' "$SYNTH"
  printf "\n${D}   … rapport synthétique complet : docs/examples/rapport_exemple.md${R}\n"
else
  printf "${Y}   (docs/examples/rapport_exemple.md introuvable)${R}\n"
fi

# ── Fin ────────────────────────────────────────────────────────────────────────
printf "\n${G}Terminé ✓${R}\n"
printf "${Y}Démo illustrative — chiffres synthétiques ; vrai audit = local avec données sous DUA.${R}\n"
