"""Tests des SCHÉMAS Pydantic = on prouve que les garde-fous de validation tiennent.

Ces tests sont PURS (juste pydantic) : ils ne démarrent ni l'app, ni le modèle, ni
MCP -> rapides et robustes en CI.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import AuditRequest


def test_defaults_are_valid():
    r = AuditRequest()
    assert r.model_name == "PaPaGei-S"
    assert r.target == "spo2"
    assert r.n_boot == 500


def test_n_boot_below_min_is_rejected():
    with pytest.raises(ValidationError):
        AuditRequest(n_boot=99)        # borne basse = 100


def test_n_boot_above_max_is_rejected():
    with pytest.raises(ValidationError):
        AuditRequest(n_boot=10_000)    # borne haute = 2000 (garde-fou ressources)


def test_unknown_model_is_rejected():
    with pytest.raises(ValidationError):
        AuditRequest(model_name="GPT-PPG")   # hors liste blanche Literal


def test_bad_target_is_rejected():
    with pytest.raises(ValidationError):
        AuditRequest(target="o2sat")          # hors {spo2, sao2}
