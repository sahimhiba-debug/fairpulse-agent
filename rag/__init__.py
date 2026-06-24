"""RAG (couche 3) — donne à l'agent une base documentaire qu'il interroge pour
CONTEXTUALISER ses résultats chiffrés (au lieu de ne sortir que des nombres).

Pipeline : docs -> chunks -> embeddings (MPS) -> index vectoriel local -> recherche
cosinus. Corpus = docs FairPulse locaux (+ dossier corpus/ extensible). Aucun
article sous copyright n'est téléchargé : seul le MODÈLE d'embeddings l'est (open).
"""
