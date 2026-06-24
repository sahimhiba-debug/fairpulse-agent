"""CONSTRUCTION DU GRAPHE — le cœur LangGraph.

Ici on assemble les 5 nœuds en une machine à états exécutable. Les 5 gestes
fondamentaux de LangGraph (à retenir) :

  1. StateGraph(AuditState)  -> crée un graphe typé par notre état partagé.
  2. add_node(nom, fonction) -> enregistre une étape (une fonction state -> maj).
  3. add_edge(a, b)          -> « après a, exécute b » : c'est le FLUX.
                                 START = point d'entrée, END = point de sortie.
  4. compile()               -> fige le graphe en une app exécutable (.invoke()).

Notre flux est LINÉAIRE (volontaire pour la couche 1, pour voir le mécanisme nu) :

    START -> load_model -> load_data -> run_inference -> compute_metrics -> write_report -> END

Aux couches suivantes, on remplacera certains add_edge par des edges
CONDITIONNELLES (add_conditional_edges) pour brancher/boucler — la structure
(nœuds isolés, état partagé) est déjà prête pour ça.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    compute_metrics_node,
    load_data_node,
    load_model_node,
    run_inference_node,
    write_report_node,
)
from agent.state import AuditState


def build_graph():
    """Assemble et compile le graphe d'audit ; renvoie l'app exécutable."""
    # 1) Un graphe dont l'état circulant est de type AuditState.
    g = StateGraph(AuditState)

    # 2) On enregistre chaque étape sous un nom. Le nom sert de cible aux edges.
    g.add_node("load_model", load_model_node)
    g.add_node("load_data", load_data_node)
    g.add_node("run_inference", run_inference_node)
    g.add_node("compute_metrics", compute_metrics_node)
    g.add_node("write_report", write_report_node)

    # 3) Les edges décrivent l'ordre = par où passe l'état.
    g.add_edge(START, "load_model")            # point d'entrée
    g.add_edge("load_model", "load_data")
    g.add_edge("load_data", "run_inference")
    g.add_edge("run_inference", "compute_metrics")
    g.add_edge("compute_metrics", "write_report")
    g.add_edge("write_report", END)            # fin

    # 4) compile() -> objet avec .invoke(state_initial) qui exécute tout le flux.
    return g.compile()
