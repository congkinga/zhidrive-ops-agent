"""Operational analytics helpers for the AI product operations layer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .storage import load_cases, load_model_logs


ROOT = Path(__file__).resolve().parent.parent
OPS_DATA_DIR = ROOT / "data" / "ops"


def read_ops_json(name: str, default: Any) -> Any:
    path = OPS_DATA_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def top_scenarios(limit: int = 5) -> list[dict[str, Any]]:
    cases = load_cases()
    counter: Counter[str] = Counter()
    for item in cases:
        scenario = (item.get("analysis") or {}).get("scenario", "待补充")
        if scenario:
            counter[scenario] += 1
    return [{"scenario": key, "count": value} for key, value in counter.most_common(limit)]


def dimension_distribution() -> list[dict[str, Any]]:
    cases = load_cases()
    counter: Counter[str] = Counter()
    for item in cases:
        dimensions = (item.get("analysis") or {}).get("dimensions") or []
        for dimension in dimensions:
            counter[dimension] += 1
    return [{"dimension": key, "count": value} for key, value in counter.most_common()]


def get_overview() -> dict[str, Any]:
    cases = load_cases()
    logs = load_model_logs(200)
    content = read_ops_json("content.json", {})
    content_items = content.get("content", [])
    return {
        "feedback_processed": len(cases),
        "closure_rate": 0.73,
        "avg_closure_hours": 18.4,
        "active_users": 48,
        "weekly_active_rate": 0.36,
        "content_interactions": sum(item.get("interactions", 0) for item in content_items),
        "model_calls": len(logs),
        "top_scenarios": top_scenarios(),
        "dimension_distribution": dimension_distribution(),
    }


def get_funnel() -> dict[str, Any]:
    return read_ops_json("funnel.json", {"funnel": []})


def get_segments() -> dict[str, Any]:
    return read_ops_json("users.json", {"segments": [], "profiles": []})


def get_activities() -> dict[str, Any]:
    return read_ops_json("activities.json", {"activities": []})


def get_content() -> dict[str, Any]:
    return read_ops_json("content.json", {"content": []})


def get_research() -> dict[str, Any]:
    return read_ops_json("research.json", {"interviews": 0, "questionnaires": 0, "insights": []})
