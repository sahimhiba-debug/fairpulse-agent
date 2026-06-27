# Captures visuelles — guide (Swagger + GIF)

Ce guide te donne les **commandes exactes** pour produire les deux visuels qui restent à ta main
(ils sont *interactifs* / *visuels*, donc à toi de les capturer). Les schémas (RAG, LangGraph),
les badges, le rapport d'exemple et `demo.sh` sont déjà prêts dans le dépôt.

Prérequis communs (locaux, jamais committés) :
```bash
export FAIRPULSE_ROOT=/chemin/vers/fairpulse     # ton checkout FairPulse
# + poids PPG téléchargés, + données OpenOximetry stagées (restreintes, DUA)
```

---

## A. Swagger (capture d'écran de l'API)

### 1. Lancer l'API en local
```bash
cd fairpulse-agent
FAIRPULSE_ROOT=$FAIRPULSE_ROOT \
PYTHONPATH=.:src \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```
> Au démarrage, l'API ouvre **une** session MCP et charge le modèle PPG une seule fois (sur MPS).
> Attends la ligne `Application startup complete.` avant de capturer.

### 2. Ouvrir Swagger UI
```
http://127.0.0.1:8000/docs
```
(ReDoc alternatif : `http://127.0.0.1:8000/redoc`)

### 3. Quoi capturer (suggestion)
- la page `/docs` avec les endpoints dépliés : **POST `/audit`**, **GET `/audit/{job_id}`**, **GET `/health`** ;
- un **« Try it out »** sur `POST /audit` avec le corps :
  ```json
  { "model_name": "PaPaGei-S", "target": "spo2" }
  ```
  → réponse **202** avec un `job_id` ;
- puis `GET /audit/{job_id}` montrant le statut (`queued` → `running` → `done` + rapport).

### 4. Où enregistrer l'image
Enregistre la capture sous **`docs/images/swagger.png`**. Le README l'affichera si tu ajoutes
(par ex. dans la section « L'architecture ») :
```markdown
<p align="center"><img src="docs/images/swagger.png" alt="API FairPulse — Swagger UI" width="760"></p>
```

---

## B. GIF de terminal (la démo de bout en bout)

> 🔴 **À FILMER : `demo_public.sh` — JAMAIS `demo.sh`.**
> `demo.sh` lance un **vrai audit** et affiche un rapport **calculé sur les données OpenOximetry
> (restreintes, DUA PhysioNet)** : le filmer **exposerait des données sous licence restreinte**
> dans un GIF public. Il reste utile **en local uniquement**.
> `demo_public.sh` est la version **vitrine** : autonome, sans données réelles, sans réseau, sans
> `FAIRPULSE_ROOT` ni poids ; il **narre** le flux des 8 nœuds (couche 3) et n'affiche, comme
> aperçu chiffré, que le rapport **synthétique étiqueté** (`docs/examples/rapport_exemple.md`).

On enregistre l'exécution de **`demo_public.sh`** avec **asciinema** (enregistrement) + **agg**
(rendu GIF).

### 1. Installer les outils (une fois)
```bash
brew install asciinema     # enregistreur de terminal
brew install agg           # asciinema-cast -> GIF   (sinon : cargo install --git https://github.com/asciinema/agg)
```

### 2. Enregistrer la démo (script PUBLIC)
```bash
cd fairpulse-agent
# Astuce : fenêtre ~100x32, police lisible, thème sombre -> GIF net.
asciinema rec demo.cast --overwrite --command "./demo_public.sh"
```
> `demo_public.sh` dure ~15-25 s et s'arrête tout seul (« Terminé ✓ »).
> Rythme réglable : `PAUSE=0.8 ./demo_public.sh` (pauses plus courtes) — mais pour `asciinema rec`
> via `--command`, fixe-le avant : `PAUSE=0.8 asciinema rec demo.cast --overwrite --command "./demo_public.sh"`.

### 3. Convertir en GIF
```bash
agg --font-size 18 --theme monokai demo.cast docs/images/demo.gif
```
(options utiles : `--speed 1.4` pour accélérer, `--cols 100 --rows 32` pour cadrer.)

### 4. Intégrer au README
```markdown
<p align="center"><img src="docs/images/demo.gif" alt="Démo terminal — flux d'audit FairPulse (illustratif)" width="760"></p>
```

> Ne committe **pas** `demo.cast` (intermédiaire) — seulement `docs/images/demo.gif`.

---

## Récap — ce qui est DÉJÀ prêt (généré automatiquement)
- `docs/images/rag_schema.png` (+ `.mmd`) — schéma RAG fidèle au code (extractif, sans LLM)
- `docs/images/langgraph.png` (+ `.mmd`) — graphe LangGraph (couche 3, 8 nœuds)
- `docs/examples/rapport_exemple.md` — exemple **synthétique** (étiqueté) du rapport
- badges README (statut + stack) · images intégrées au README
- `demo_public.sh` — démo **vitrine** autonome (à filmer pour le GIF) · `demo.sh` — démo **locale** (vrai audit, NE PAS filmer)
