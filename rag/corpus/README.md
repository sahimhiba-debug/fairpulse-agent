# Corpus RAG — dépose ici TES sources additionnelles

Ce dossier est un **drop-folder** : tout fichier `.md` ou `.txt` que tu y places
sera intégré au corpus RAG au prochain `python -m rag.build_index`.

## Règles (garde-fous)
- ⚠️ **Pas d'articles sous copyright en intégral.** Mets uniquement :
  - tes propres notes / résumés,
  - des **métadonnées + résumés** que tu rédiges (titre, auteurs, année, DOI, 2-3 phrases),
  - des extraits de **sources ouvertes vérifiées** (licence permettant la réutilisation).
- Garde une **citation claire** en tête de chaque fichier (titre, source, URL/DOI) pour que
  le RAG puisse l'attribuer correctement.

## Format conseillé (une source = un fichier)
```markdown
# Titre court de la source
Source : Auteur(s), Revue/Éditeur, Année. DOI/URL.

Résumé / notes en quelques paragraphes (tes mots ou extrait sous licence ouverte).
Points clés pertinents pour l'audit FairPulse (biais d'oxymétrie, équité PPG, …).
```

> Note : tes références curées sont **déjà** dans `docs/rapport_v1.md` (§5.5–5.6),
> donc le RAG fonctionne sans rien ajouter ici. Ce dossier sert à l'enrichir.
