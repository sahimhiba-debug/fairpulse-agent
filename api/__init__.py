"""API FastAPI (couche 4) — expose l'agent d'audit FairPulse comme un service.

PALIER 1 : endpoints + jobs de fond EN MÉMOIRE + observabilité + guardrails.
Le modèle PPG vit dans le serveur MCP (chargé une fois sur MPS) ; l'API ouvre
une session MCP persistante au démarrage et la réutilise pour chaque requête.
"""
