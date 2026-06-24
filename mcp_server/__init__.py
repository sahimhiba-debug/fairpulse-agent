"""Serveur MCP FairPulse (couche 2) — expose l'audit comme des tools MCP.

Le serveur DÉTIENT l'état lourd (modèle torch sur MPS, embeddings) ; les clients
n'échangent que des *handles* (chaînes) + métadonnées JSON. Voir store.py.
"""
