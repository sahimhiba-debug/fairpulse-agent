"""Configuration de l'API via VARIABLES D'ENVIRONNEMENT (jamais en dur).

Pourquoi pas de valeurs en dur ? (1) Sécurité : un secret en dur finit dans git,
impossible à révoquer proprement. (2) Portabilité : la même image tourne en local,
en CI, en prod avec des valeurs différentes — sans recompiler. (3) Rotation : on
change une variable d'env, pas le code.

pydantic-settings lit automatiquement, par ordre de priorité : variables d'env >
fichier .env (git-ignored) > défauts ci-dessous. Chaque champ est typé et validé.
Au PALIER 1 il n'y a pas encore de secret (Postgres viendra au palier 2 avec
DATABASE_URL) ; on met en place le mécanisme proprement dès maintenant.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file=".env" : lit un .env local s'il existe ; extra="ignore" tolère
    # d'autres variables (ex. HF_TOKEN du projet) sans planter.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    api_host: str = "127.0.0.1"      # localhost uniquement (pas d'exposition réseau)
    api_port: int = 8000

    # Garde-fou ressources : nombre max de jobs en attente dans la file.
    # Au-delà, POST /audit renvoie 429 (file pleine) au lieu de saturer la machine.
    max_queue: int = 4

    # Device d'inférence : "auto" -> MPS si dispo (résolu côté serveur MCP).
    device: str = "auto"

    # PALIER 2 — persistance. Si DATABASE_URL est défini -> jobs en PostgreSQL ;
    # sinon -> jobs en mémoire (palier 1). Le secret (mot de passe) vit UNIQUEMENT
    # dans cette variable d'env (jamais en dur dans le code).
    #   ex : postgresql+psycopg://user:password@localhost:5432/fairpulse
    database_url: str | None = None


settings = Settings()
