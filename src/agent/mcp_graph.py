"""Graphe COUCHE 2 — même topologie qu'en couche 1, mais 6 nœuds appellent des tools MCP.

La structure LangGraph est identique (StateGraph / add_node / add_edge / compile) :
seul le CONTENU des nœuds change (appels MCP au lieu d'appels directs). C'est la
preuve que la couche 1 était bien découplée — on a pu glisser MCP dessous sans
toucher à la forme du graphe.

    START → load_model → load_data → run_inference → compute_metrics
          → run_calibration_test → run_robustness_test → write_report → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.mcp_nodes import (
    calibration_node,
    compute_metrics_node,
    load_data_node,
    load_model_node,
    robustness_node,
    run_inference_node,
    write_report_node,
)
from agent.mcp_state import McpAuditState


def build_mcp_graph():
    g = StateGraph(McpAuditState)

    g.add_node("load_model", load_model_node)
    g.add_node("load_data", load_data_node)
    g.add_node("run_inference", run_inference_node)
    g.add_node("compute_metrics", compute_metrics_node)
    g.add_node("run_calibration_test", calibration_node)
    g.add_node("run_robustness_test", robustness_node)
    g.add_node("write_report", write_report_node)

    g.add_edge(START, "load_model")
    g.add_edge("load_model", "load_data")
    g.add_edge("load_data", "run_inference")
    g.add_edge("run_inference", "compute_metrics")
    g.add_edge("compute_metrics", "run_calibration_test")
    g.add_edge("run_calibration_test", "run_robustness_test")
    g.add_edge("run_robustness_test", "write_report")
    g.add_edge("write_report", END)

    return g.compile()
