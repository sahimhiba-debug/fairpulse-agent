> # ⚠️ EXEMPLE SYNTHÉTIQUE — chiffres illustratifs, NON issus d'un audit réel
>
> Ce document montre **uniquement la STRUCTURE** du rapport produit par l'agent
> (mise en page identique à `src/agent/report.py:render_markdown`). **Tous les chiffres
> ci-dessous sont inventés** à des fins d'illustration. Ils ne proviennent **d'aucun audit**
> et ne décrivent **aucun modèle réel**. Aucune donnée OpenOximetry (restreinte, DUA PhysioNet)
> n'a été lue, exécutée ou exposée pour générer cet exemple.
>
> Pour obtenir un **vrai** rapport, il faut un checkout FairPulse local + les poids + les données
> restreintes (cf. README §« Prérequis ») et lancer `python -m agent.run_rag`.

---

# Rapport d'audit FairPulse — PaPaGei-S

- **Modèle audité** : PaPaGei-S  ·  **device** : `mps`
- **Données** : OpenOximetry (local, restreint — audit hors-ligne)  ·  **tâche** : spo2
- **Protocole** : LOSO (leave-one-subject-out), IC bootstrap niveau patient  ·  **n_boot** : 500  ·  **seed** : 0
- **Couverture** : **312** fenêtres / **48** patients

## Performance globale
- **MAE** (%SpO2), IC95 bootstrap (niveau patient) : **2.41 [2.08, 2.79]**
- RMSE : 3.27  ·  Pearson r : 0.812

## Équité par teint de peau (Monk) + calibration
_Gap de disparité = **1.36 %SpO2** · groupe le plus dégradé : **dark**_

| Teint | MAE [IC95] | Biais calibration (%SpO2) | LOSTO MAE [IC95] |
|---|---|---|---|
| light | 1.92 [1.61, 2.27] | -0.18 | 2.05 [1.70, 2.44] |
| medium | 2.38 [2.01, 2.80] | +0.41 | 2.61 [2.18, 3.09] |
| dark | 3.28 [2.74, 3.88] | +1.07 | 3.74 [3.10, 4.45] |

### Lecture
- **MAE [IC95]** : erreur intra-groupe (leave-one-subject-out), IC ré-échantillonné au niveau patient.
- **Biais calibration** : résidu signé moyen (pred − vrai). `>0` = sur-estime la saturation, `<0` = sous-estime.
- **LOSTO** : *leave-one-skin-tone-out* — erreur sur un teint **jamais vu** à l'entraînement (robustesse sous décalage de teint).

> ⚠️ OpenOximetry est une donnée restreinte (DUA). Cet audit tourne strictement en local / in-process ; ce rapport ne contient que des métriques agrégées. Label au niveau *encounter* (le PPG n'est pas time-synced à SpO2/SaO2 en v1.1.1) : valeurs absolues à interpréter comme un ordre de grandeur, pas une vérité par fenêtre.

## Contexte (littérature) — RAG
_Passages récupérés par recherche sémantique dans le corpus FairPulse local, pour contextualiser les chiffres ci-dessus. Citations = sources réelles du corpus._

- **rapport_v1.md § 5.6 (biais d'oxymétrie)**  _(similarité 0.71)_
  > Les oxymètres de pouls ont tendance à surestimer la SaO2 réelle chez les patients à peau foncée, ce qui peut masquer une hypoxémie occulte et retarder la prise en charge clinique.
- **rapport_v1.md § 5.5 (équité des foundation models PPG)**  _(similarité 0.64)_
  > L'erreur d'estimation des modèles de fondation PPG n'est pas uniforme entre teints : le groupe le plus sombre présente une MAE et un biais de calibration plus élevés, cohérents avec le biais matériel documenté.
- **rapport_v1.md § 4.2 (protocole de robustesse)**  _(similarité 0.58)_
  > Le test leave-one-skin-tone-out évalue la généralisation à un teint absent de l'entraînement ; une dégradation sous LOSTO signale une dépendance du modèle à la distribution des teints vue.

_Source de chaque passage = fichier du corpus local ci-dessus. Les références bibliographiques complètes (Sjoding 2020, Fawzy 2022, …) figurent dans `rapport_v1.md` §5.6._

---
### Trace d'exécution du graphe
```
[1/7] load_model (MCP) : PaPaGei-S sur 'mps' -> handle=mdl_0
[2/7] load_data_sample (MCP) : 312 fenêtres / 48 patients {'light': 16, 'medium': 16, 'dark': 16} -> handle=dat_0
[3/7] run_inference (MCP) : embeddings (312, 512) (MPS, côté serveur) -> handle=emb_0
[4/7] compute_metrics (MCP) : MAE globale 2.41 [2.08, 2.79]
[5/7] run_calibration_test (MCP) : gap=1.36 worst=dark
[6/7] run_robustness_test (MCP) : LOSTO sur 3 teints
[RAG] retrieve_context : 3 passages pertinents (sur 3 requêtes dérivées des résultats)
[7/7] write_report (local) : rapport écrit -> reports/audit_rag_PaPaGei-S_spo2.md
```

---
_⚠️ Rappel : exemple **synthétique**. Aucune valeur ci-dessus ne constitue un résultat de mesure._
