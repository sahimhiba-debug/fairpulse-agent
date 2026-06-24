# CI/CD — workflow GitHub Actions (dépôt dédié `fairpulse-agent`)

Le pipeline est dans [`ci.yml`](./ci.yml), avec des chemins **relatifs à la racine du dépôt
dédié** (où l'agent est à la racine). Il est **préparé ici** mais pas encore actif : GitHub
n'exécute que les workflows placés dans **`.github/workflows/`**.

## Ce que fait le pipeline (à chaque push sur `main` et chaque PR)
Deux jobs en parallèle, sur des runners **Linux (CPU, pas de MPS)** :

1. **lint-test** — `ruff check .` (lint) + `pytest api/tests/test_schemas.py` (validation Pydantic).
   *Les tests de schémas ne dépendent pas de `fairpulse` → tournent partout. Les tests
   d'endpoints, qui importent `fairpulse`, se lancent en local avec `FAIRPULSE_ROOT`.*
2. **docker-build** — `docker build` de l'image API (vérifie qu'elle se construit ; ne la lance pas).

Tout vert ✅ = lint OK, garde-fous de validation OK, image buildable.

## Pour l'ACTIVER (c'est TOI qui pousses)
```bash
mkdir -p .github/workflows
cp ci/ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "ci: lint + tests + docker build"
git push
```

## Ce qui se passe au push
GitHub détecte `.github/workflows/ci.yml`, lit le `on:`, **déclenche** le workflow, provisionne
des **runners** Linux jetables et affiche ✅/❌ dans l'onglet **Actions** (et sous chaque commit/PR).
Aucun secret requis. Si un jour tu pousses l'image vers un registre, le token ira dans
**Settings → Secrets and variables → Actions** (jamais dans le YAML).
