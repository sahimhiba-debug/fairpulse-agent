"""Graphe COUCHE 3 (RAG) — la chaîne couche 2 + un nœud retrieve_context.

Différence avec la couche 2 : on insère `retrieve_context` APRÈS les métriques et
AVANT le rapport. Le rapport ajoute alors une section « Contexte (littérature) ».
Le reste (nœuds 1-6, write_report) est réutilisé tel quel -> on ne refait rien.

    START → load_model → load_data → run_inference → compute_metrics
          → run_calibration_test → run_robustness_test
          → retrieve_context → write_report → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.mcp_nodes import (
    calibration_node,
    compute_metrics_node,
    load_data_node,
    load_model_node,
    retrieve_context_node,
    robustness_node,
    run_inference_node,
    write_report_node,
)
from agent.mcp_state import McpAuditState


def build_rag_graph():
    g = StateGraph(McpAuditState)

    g.add_node("load_model", load_model_node)
    g.add_node("load_data", load_data_node)
    g.add_node("run_inference", run_inference_node)
    g.add_node("compute_metrics", compute_metrics_node)
    g.add_node("run_calibration_test", calibration_node)
    g.add_node("run_robustness_test", robustness_node)
    g.add_node("retrieve_context", retrieve_context_node)   # ← nouveau (RAG)
    g.add_node("write_report", write_report_node)

    g.add_edge(START, "load_model")
    g.add_edge("load_model", "load_data")
    g.add_edge("load_data", "run_inference")
    g.add_edge("run_inference", "compute_metrics")
    g.add_edge("compute_metrics", "run_calibration_test")
    g.add_edge("run_calibration_test", "run_robustness_test")
    g.add_edge("run_robustness_test", "retrieve_context")   # ← inséré ici
    g.add_edge("retrieve_context", "write_report")
    g.add_edge("write_report", END)

    return g.compile()
