#!/usr/bin/env python3
"""Project runtime JSON storage helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CASES_FILE = DATA_DIR / "cases" / "cases.json"
EVAL_FILE = DATA_DIR / "eval" / "eval_cases.json"
LOG_FILE = DATA_DIR / "logs" / "model_calls.jsonl"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_cases() -> list[dict[str, Any]]:
    return read_json(CASES_FILE, [])


def save_case(case: dict[str, Any]) -> dict[str, Any]:
    cases = load_cases()
    case_id = case.get("id") or f"case-{int(time.time() * 1000)}"
    case["id"] = case_id
    case["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cases = [item for item in cases if item.get("id") != case_id]
    cases.insert(0, case)
    write_json(CASES_FILE, cases[:100])
    return case


def load_eval_cases() -> list[dict[str, Any]]:
    return read_json(EVAL_FILE, [])


def log_model_call(entry: dict[str, Any]) -> None:
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    append_jsonl(LOG_FILE, entry)


def load_model_logs(limit: int = 50) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    logs = []
    for line in reversed(lines):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(logs) >= limit:
            break
    return logs
