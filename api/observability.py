"""Observabilité — logging STRUCTURÉ et HORODATÉ.

« Observabilité » = pouvoir répondre, en production, à : *que s'est-il passé, quand,
combien de temps, et où ça a cassé ?* — sans rejouer le bug. C'est vital car en prod
on ne voit pas l'écran : les logs sont souvent la SEULE fenêtre sur le comportement.

On émet des lignes au format `clé=valeur` (faciles à grep et à parser par un
collecteur type Loki/Datadog), chacune horodatée. Chaque étape de l'audit (nœud du
graphe) loggue son nom et sa durée -> on peut suivre et profiler un audit en prod.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("fairpulse_api")
    if logger.handlers:           # évite les handlers en double (reload uvicorn)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    # ts=… level=… logger=… <message clé=valeur> : format structuré, greppable.
    handler.setFormatter(logging.Formatter(
        "ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logging()
