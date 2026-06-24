"""Configuration transverse : chemins fairpulse, device (MPS), reproductibilité.

Rien de spécifique à LangGraph ici — c'est l'environnement d'exécution que les
nœuds partagent. On le garde isolé pour qu'un nœud n'ait jamais à « deviner »
où sont les poids, les données, ni sur quel device tourner.

⚠️ Garde-fou MPS : on pose PYTORCH_ENABLE_MPS_FALLBACK=1 AVANT tout import de
torch. Certaines opérations PPG (FFT, etc.) n'ont pas de noyau Metal ; ce flag
les fait retomber sur le CPU au lieu de planter. Il DOIT être posé avant que
torch ne s'initialise — donc à l'import de ce module, qui est importé en premier.
"""

from __future__ import annotations

import os

# --- Garde-fou MPS : poser le fallback AVANT que torch ne soit importé -----------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random  # noqa: E402  (après le setdefault ci-dessus, volontairement)
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# --- Localisation du projet fairpulse (lecture seule) ----------------------------
# config.py est à .../fairpulse/fairpulse_agent/src/agent/config.py
#   parents[0]=agent  [1]=src  [2]=fairpulse_agent  [3]=<racine fairpulse>
# Racine du projet FairPulse (fournit le code `fairpulse`, les poids, les données et
# le corpus RAG). Par défaut : déduite de l'emplacement (agent DANS le repo FairPulse).
# Pour un dépôt DÉDIÉ (agent seul), pointe FAIRPULSE_ROOT vers ton checkout FairPulse
# local via la variable d'environnement -> aucun fichier parent n'est embarqué.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
FAIRPULSE_ROOT = (Path(os.environ["FAIRPULSE_ROOT"]).resolve()
                  if os.environ.get("FAIRPULSE_ROOT") else _DEFAULT_ROOT)
WEIGHTS_DIR = FAIRPULSE_ROOT / "models" / "weights"          # poids PaPaGei-S / Pulse-PPG
OPENOX_ROOT = FAIRPULSE_ROOT / "data" / "raw" / "openox-repo"  # données restreintes, LOCALES

# Sorties de l'agent — dossier dédié, gitignored, jamais envoyé en ligne.
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def resolve_device(requested: str = "auto") -> str:
    """Choisir le device torch. 'auto' -> MPS (GPU Apple) si dispo, sinon CPU.

    On importe torch ICI (et pas en tête de module) pour que `config` reste
    importable même sans l'extra `deep` — utile aux couches futures qui ne font
    pas d'inférence.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = 0) -> None:
    """Fixer toutes les sources d'aléa pour un run reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
