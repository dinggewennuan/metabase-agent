from __future__ import annotations

from typing import Any


def rank_metrics(search_result: dict[str, Any], business_terms: list[str]) -> list[dict[str, Any]]:
    rows = search_result.get("data", []) if isinstance(search_result, dict) else []
    metrics = [row for row in rows if row.get("type") == "metric"]

    def score(metric: dict[str, Any]) -> tuple[int, str]:
        name = str(metric.get("name") or metric.get("display_name") or "").lower()
        description = str(metric.get("description") or "").lower()
        verified = 2 if metric.get("verified") else 0
        term_score = sum(3 for term in business_terms if term.replace("_", " ") in name or term in description)
        return verified + term_score, name

    return sorted(metrics, key=score, reverse=True)


def choose_metric(search_result: dict[str, Any], business_terms: list[str]) -> dict[str, Any] | None:
    ranked = rank_metrics(search_result, business_terms)
    return ranked[0] if ranked else None
