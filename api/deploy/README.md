# Déploiement local — PALIER 2 (PostgreSQL)

Persistance des jobs d'audit en Postgres. **Toi** tu lances la base et tu fixes les
secrets dans `.env` ; le code ne contient **aucun** identifiant en dur.

## 1. Configure ton `.env` (jamais commité)
Copie l'exemple puis choisis ton mot de passe :
```bash
cp fairpulse_agent/api/.env.example fairpulse_agent/.env
# édite fairpulse_agent/.env : remplace 'change-me-...' par TON mot de passe,
# dans POSTGRES_PASSWORD ET dans DATABASE_URL (le même).
```

## 2. Lance Postgres (au choix)

**Option A — Docker (recommandé, portable)**
```bash
docker compose --env-file fairpulse_agent/.env \
    -f fairpulse_agent/api/deploy/docker-compose.db.yml up -d
```

**Option B — Postgres.app (déjà installé chez toi)**
Démarre Postgres.app, puis crée la base/role correspondant à ton `.env` :
```bash
createuser -s fairpulse        # si besoin
createdb -O fairpulse fairpulse
# (mot de passe : ALTER USER fairpulse PASSWORD '...' pour matcher DATABASE_URL)
```

## 3. Lance l'API (elle lit DATABASE_URL depuis l'env)
```bash
cd /Users/hiba/Desktop/fairpulse
set -a; source fairpulse_agent/.env; set +a        # charge .env dans l'environnement
PYTHONPATH=fairpulse_agent:fairpulse_agent/src PYTORCH_ENABLE_MPS_FALLBACK=1 \
  fairpulse_agent/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Au démarrage tu verras dans les logs : `event=store backend=sql scheme=postgresql+psycopg`.

## 4. Vérifie la persistance (jobs survivent au redémarrage)
1. `POST /audit` → récupère un `job_id`, attends `done`.
2. **Arrête** l'API (Ctrl-C) puis **relance**-la (étape 3).
3. `GET /audit/{job_id}` sur l'API fraîchement redémarrée → le job est **toujours là**
   (il vient de Postgres, pas de la mémoire). Tu peux aussi le voir en SQL :
   ```bash
   psql "$DATABASE_URL" -c "SELECT job_id, status, created_at FROM audit_jobs;"
   ```

> Note prod : ici on crée la table via `Base.metadata.create_all`. En vrai projet,
> on gérerait le schéma avec des **migrations** (Alembic) pour versionner les changements.
