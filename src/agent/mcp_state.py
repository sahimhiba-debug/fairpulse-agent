"""État partagé de l'agent COUCHE 2 (MCP).

Différence clé avec la couche 1 : l'état ne porte plus d'objets lourds (modèle,
embeddings). Le serveur MCP les détient ; l'agent ne transporte que des *handles*
(chaînes) et des dicts JSON renvoyés par les tools. C'est plus léger et sérialisable.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class McpAuditState(TypedDict, total=False):
    config: dict[str, Any]      # entrée : model_name, target, device, n_boot, seed, tailles

    model_info: dict[str, Any]  # nœud 1 : {model_handle, name, device, weights_path}
    data_info: dict[str, Any]   # nœud 2 : {data_handle, n_windows, n_patients, per_group}
    emb_info: dict[str, Any]    # nœud 3 : {emb_handle, shape}
    metrics: dict[str, Any]     # nœud 4 : sortie de compute_metrics
    calibration: dict[str, Any] # nœud 5 : sortie de run_calibration_test
    robustness: dict[str, Any]  # nœud 6 : sortie de run_robustness_test
    context: dict[str, Any]     # nœud RAG (couche 3) : {queries:[...], passages:[...]}

    report_md: str              # nœud 7 (local)
    report_path: str

    log: Annotated[list[str], operator.add]  # reducer : s'accumule (cf. couche 1)
