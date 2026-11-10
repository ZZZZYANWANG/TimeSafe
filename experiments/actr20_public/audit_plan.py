"""Lightweight manifest shared by ACTR audit workers and orchestrators."""

from __future__ import annotations

AUDIT_NAMES = (
    "mature_feedback_prefix",
    "legal_refit",
    "full_sample_negative_control",
    "legal_availability",
    "immature_label_negative_control",
    "refit_determinism",
    "label_maturity_and_purge",
)

EXPECTED = {
    "mature_feedback_prefix": {"PASS"},
    "legal_refit": {"PASS", "WARN"},
    "full_sample_negative_control": {"FAIL"},
    "legal_availability": {"PASS", "WARN"},
    "immature_label_negative_control": {"FAIL"},
    "refit_determinism": {"PASS", "WARN"},
    "label_maturity_and_purge": {"PASS"},
}
