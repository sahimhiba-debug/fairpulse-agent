# FairPulse Agent — un agent d'audit ML, construit par couches

Un **agent d'audit** qui mesure la **robustesse et l'équité** des *foundation models* PPG
(photopléthysmographie / oxymétrie) **selon la couleur de peau** — le biais d'oxymétrie étant
un problème de sécurité patient documenté (un oxymètre peut surestimer la saturation sur peau
foncée, retardant la prise en charge). L'agent **orchestre** un pipeline d'audit existant — le
projet **FairPulse** (suite de benchmark indépendante pour l'IA des signaux physiologiques) — et
le transforme, couche après couche, en un **service déployable**.

> Il n'entraîne aucun modèle et ne redécouvre aucun biais : il **audite** des modèles ouverts
> existants (PaPaGei-S, Pulse-PPG) sur un protocole rigoureux, et **contextualise** ses chiffres
> avec la littérature. Les modèles sont des *sujets* d'audit, pas des concurrents.

Ce dépôt est un **projet pédagogique** autant qu'un produit : chaque couche apprend une brique
de l'ingénierie d'un agent ML, et le code est commenté pour être **défendable**.

---

## L'architecture en 5 couches

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Couche 4 — PRODUCTION                                                     │
  │    FastAPI (jobs async) → PostgreSQL (persistance) → Docker → CI/CD        │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  Couche 3 — RAG : contextualise le rapport avec la littérature        │  │
  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │
  │  │  │  Couche 2 — MCP : les outils d'audit exposés en serveur MCP     │  │  │
  │  │  │  ┌─────────────────────────────────────────────────────────┐  │  │  │
  │  │  │  │  Couche 1 — LangGraph : l'audit en 5 nœuds (état partagé) │  │  │  │
  │  │  │  └─────────────────────────────────────────────────────────┘  │  │  │
  │  │  └───────────────────────────────────────────────────────────────┘  │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘
         réutilise (lecture seule) le code d'audit du projet FairPulse
```

| Couche | Rôle | Brique apprise |
|---|---|---|
| **1 · LangGraph** | L'audit en **5 nœuds** explicites (charge modèle → charge données → inférence → métriques+IC → rapport), avec un **état partagé** qui circule. | orchestration par graphe d'états |
| **2 · MCP** | Les opérations d'audit exposées comme **outils** d'un serveur **MCP** (Model Context Protocol) ; l'agent les appelle via le protocole. Réutilisables par **tout** client MCP (Claude Desktop…). | protocoles d'outils, découplage |
| **3 · RAG** | L'agent interroge une base documentaire locale (embeddings + index vectoriel) et **cite** des passages réels pour contextualiser ses chiffres. | retrieval sémantique, ancrage |
| **4 · API + Prod** | Une **API FastAPI** (jobs asynchrones), persistance **PostgreSQL**, **Docker**, **CI/CD** GitHub Actions. | mise en production |

Le rapport produit est **identique** d'une couche à l'autre (mêmes chiffres) — chaque couche
ajoute une capacité d'**infrastructure**, pas un changement de résultat. Le rapport contient :
MAE par groupe de teint **avec IC bootstrap au niveau patient**, biais de calibration, disparité
(*gap*), test de robustesse **leave-one-skin-tone-out**, puis le **contexte littérature** sourcé.

---

## Prérequis pour exécuter ⚠️

Ce dépôt contient **uniquement l'agent**. Pour qu'il **tourne**, il a besoin, fournis
**localement** (jamais embarqués ici pour des raisons de licence/taille) :

- le code d'audit **FairPulse** (`src/fairpulse`) — code MIT, hébergé dans le projet FairPulse ;
- les **poids** des modèles PPG (PaPaGei-S / Pulse-PPG), téléchargés par toi ;
- les **données** OpenOximetry (licence DUA PhysioNet) — **restreintes, locales uniquement** ;
- le **corpus RAG** = les docs FairPulse (`docs/*.md`), lus à la construction de l'index.

Tu pointes ton checkout FairPulse local via la variable d'environnement **`FAIRPULSE_ROOT`** :
```bash
export FAIRPULSE_ROOT=/chemin/vers/fairpulse
```
Sans FairPulse local, ce dépôt reste une **vitrine de code/architecture** parfaitement lisible
(lint + tests de schémas + build Docker passent en CI sans lui).

## Démarrage rapide

**Outils** : Python ≥ 3.11, [`uv`](https://github.com/astral-sh/uv), et (localement) un Mac
Apple Silicon pour l'accélération **MPS** (sinon CPU).

```bash
# 1. Environnement + dépendances de l'agent (+ fairpulse[deep] qui apporte torch)
uv venv .venv
uv pip install --python .venv -e . -e "$FAIRPULSE_ROOT[deep]"

# 2. (Couche 3) Construire l'index RAG une fois (lit le corpus dans $FAIRPULSE_ROOT/docs)
PYTHONPATH=.:src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m rag.build_index
```

**Lancer l'audit** — au choix selon la couche :

```bash
ENV="PYTHONPATH=.:src PYTORCH_ENABLE_MPS_FALLBACK=1 FAIRPULSE_ROOT=$FAIRPULSE_ROOT"
PY=".venv/bin/python"

env $ENV $PY -m agent.run        --model PaPaGei-S --target spo2   # Couche 1 (LangGraph direct)
env $ENV $PY -m agent.run_mcp    --model PaPaGei-S --target spo2   # Couche 2 (via serveur MCP)
env $ENV $PY -m agent.run_rag    --model PaPaGei-S --target spo2   # Couche 3 (+ contexte RAG)
```

**Couche 4 — le service** :

```bash
# API en local (modèle chargé une fois au démarrage, sur MPS)
env $ENV .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/audit -H 'Content-Type: application/json' \
     -d '{"model_name":"PaPaGei-S","target":"spo2"}'      # -> {job_id}
curl http://127.0.0.1:8000/audit/<job_id>                  # -> statut + rapport

# Ou tout en conteneurs (API + Postgres) — voir api/deploy/
cp api/.env.example .env     # choisis ton mot de passe + renseigne FAIRPULSE_SRC/WEIGHTS/DATA
docker compose --env-file .env -f api/deploy/docker-compose.yml up --build
```

---

## Trois points techniques défendables

1. **Découplage par interface stable.** Du palier 1 (jobs en mémoire) au palier 2 (PostgreSQL),
   on n'a **pas touché une ligne** des endpoints : le `JobStore` mémoire et le `SqlJobStore`
   exposent la **même interface** (`create / get / mark_*`), sélectionnés par `DATABASE_URL`.
   Même principe entre couche 1 (appels directs) et couche 2 (outils MCP) : **même graphe**,
   contenu des nœuds seul changé. *Le code dépend d'une interface, pas d'une implémentation.*

2. **La limite MPS en conteneur — assumée, pas masquée.** En local, l'inférence tourne sur le
   GPU Apple (**MPS**). Dans un conteneur Docker (VM **Linux**), MPS n'existe pas — Metal est
   propre à macOS et n'est pas exposable à un conteneur (contrairement aux GPU NVIDIA/CUDA).
   Le code le gère **tout seul** (`resolve_device("auto")` → `cpu`) : ce n'est pas un bug mais
   une **limite structurelle** Docker/Apple Silicon. Conséquence : plus lent en conteneur ;
   en prod réelle on viserait des runners GPU NVIDIA.

3. **RAG anti-hallucination.** Le rapport ne cite **que** des passages réellement récupérés du
   corpus (texte + source + score de similarité). Si rien ne dépasse le seuil, l'agent écrit
   *« pas de contexte documentaire pertinent »* — il **n'invente jamais** de référence. Vérifié :
   une requête hors-sujet renvoie **zéro** passage.

Autres choix défendables : **splits au niveau patient** (jamais segment/fenêtre), **IC bootstrap
sujet** (les fenêtres d'un patient sont corrélées), **validation Pydantic stricte** comme garde-fou,
**secrets via variables d'environnement** (jamais en dur), **observabilité** (un log horodaté + une
durée par nœud).

---

## Éthique des données (garde-fous)

- **Données OpenOximetry = restreintes (DUA PhysioNet).** Elles restent **strictement locales**,
  l'audit tourne **en-process / hors-ligne**, et le rapport ne contient **que des agrégats**
  (jamais de signal brut ni d'identifiant patient). **Jamais** versionnées ni envoyées en ligne.
- **Poids des modèles** : non redistribués (licences + taille) — téléchargés localement par l'utilisateur.
- Le `.gitignore` exclut données, poids, `.env` (secrets) et environnements virtuels.
- Le code du projet FairPulse (`../src/fairpulse`) est **réutilisé en lecture seule**, jamais modifié.

---

## Organisation du code

```
fairpulse_agent/
├── src/agent/        # couches 1-3 : graphes LangGraph, nœuds, client MCP, état, rendu rapport
│   ├── graph.py · nodes.py · state.py            (couche 1)
│   ├── mcp_*.py · run_mcp.py                      (couche 2)
│   └── mcp_graph_rag.py · run_rag.py             (couche 3)
├── mcp_server/       # couche 2 : serveur MCP (tools + resource + prompt) + registre stateful
├── rag/              # couche 3 : chunker · embedder (MPS) · store (cosinus) · build_index · corpus/
├── api/              # couche 4 : FastAPI · schemas · jobs · db (Postgres) · observability
│   ├── deploy/       #   Dockerfile · docker-compose · requirements
│   └── tests/        #   tests (validation + endpoints)
└── ci/               # couche 4 : workflow GitHub Actions (à copier vers .github/workflows/)
```

**Tests** : `PYTHONPATH=.:src pytest api/tests` (les tests d'endpoints requièrent FairPulse local ;
les tests de schémas tournent partout) · **Lint** : `ruff check .`. Le pipeline CI lance le lint,
les tests de schémas et le build de l'image Docker.

---

## Licence

Distribué sous licence **MIT** — voir [`LICENSE`](./LICENSE). © 2026 Hiba Sahim Elmir.

S'appuie sur le projet **FairPulse** (suite de benchmark indépendante pour la robustesse et
l'équité de l'IA des signaux physiologiques), réutilisé en lecture seule.
