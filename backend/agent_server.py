#!/usr/bin/env python3
"""智驾运营 Agent 本地服务。

提供静态页面和一个 /api/chat 接口。
没有配置大模型 API Key 时，使用本地文档检索和规则回答；
配置 OPENAI_API_KEY、DEEPSEEK_API_KEY 或兼容接口后，自动调用大模型。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .rag_engine import RAGEngine
from .vector_rag import VectorRAG
from .storage import (
    load_cases,
    load_eval_cases,
    load_model_logs,
    log_model_call,
    save_case,
)


ROOT = Path(__file__).resolve().parent.parent

DOC_FILES = [
    "SOUL.md",
    "README.md",
    "CASE_STUDY.md",
    "product/README.md",
    "docs/zhijia-industry-observation.md",
    "docs/agent-architecture.md",
]

RAG = RAGEngine(ROOT, DOC_FILES)
VECTOR_RAG: VectorRAG | None = None


def active_rag() -> Any:
    if VECTOR_RAG is not None and VECTOR_RAG.available:
        return VECTOR_RAG
    return RAG


def rag_context(query: str) -> str:
    return active_rag().build_context(query)


def rag_sources(query: str) -> list[str]:
    return active_rag().sources(query)


def rag_search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return active_rag().search(query, top_k=top_k)


def load_dotenv() -> None:
    """Read project-local .env.local and .env without overriding existing env."""
    for name in (".env.local", ".env"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def local_answer(query: str) -> str:
    lowered = query.lower()
    if re.search(r"五维|分类|归因", query):
        return (
            "五个维度是感知、决策、交互、硬件适配、场景边界。"
            "先判断问题属于哪个方向，再决定责任方和验证方式。"
        )
    if re.search(r"做什么|是什么|干嘛|介绍", query):
        return (
            "这个项目是智驾产品体验运营助手。"
            "它把智驾体验反馈转成场景、现象、证据、问题维度、责任方和闭环动作，"
            "用于展示 NOA 场景评测、问题归因和跨团队反馈闭环能力。"
        )
    if re.search(r"怎么用|操作|使用|打开", query):
        return (
            "打开 product/landing/zhijia-nova-ops.html，选择场景，填写现象，"
            "选择严重度和问题维度，再点击生成问题记录。"
            "也可以直接打开 product/landing/zhijia-agent.html 用聊天方式操作。"
        )
    if re.search(r"竞品|行业|趋势|公开", query):
        return (
            "公开观察主要看城市 NOA 使用率、园区和环岛等场景降级、"
            "用户投诉中的感知与交互问题，以及行业资料中的体验闭环需求。"
        )
    if re.search(r"接入|大模型|api|llm|gpt|claude|deepseek|qwen", lowered):
        return (
            "当前服务支持可选大模型。设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY 后，"
            "会自动调用对应模型；没有设置时使用本地文档检索回答。"
        )
    if re.search(r"责任方|闭环|复盘", query):
        return (
            "闭环模板包括问题描述、判断依据、短期动作、验证方式、复盘结论。"
            "这样反馈不会停在主观体验描述。"
        )
    context = rag_context(query)
    return "根据项目资料，找到以下相关说明：\n\n" + context


def llm_endpoint() -> tuple[str, str] | None:
    if os.getenv("OPENAI_API_KEY"):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        return f"{base}/chat/completions", model
    if os.getenv("DEEPSEEK_API_KEY"):
        base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return f"{base}/chat/completions", model
    return None


def call_llm_request(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return call_llm_messages(messages, temperature=temperature, json_mode=json_mode)


def call_llm_messages(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str | None:
    started = time.time()
    configured = llm_endpoint()
    if not configured:
        return None
    endpoint, model = configured
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
        data = json.loads(raw)
    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None
    log_model_call(
        {
            "model": model,
            "mode": "llm",
            "latency_ms": round((time.time() - started) * 1000, 2),
            "user_content_chars": sum(len(item.get("content", "")) for item in messages),
            "reply_chars": len(reply),
            "json_mode": json_mode,
        }
    )
    return reply


def call_llm(query: str, context: str, history: list[dict[str, str]] | None = None) -> str | None:
    system_prompt = (
        "你是智驾产品体验运营助手。请根据提供的项目资料回答用户问题。"
        "回答要准确、简洁、适合中国智能驾驶产品运营场景。不要编造资料中没有的事实。"
        "如果资料不足，请说明无法从当前项目资料确认。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if history:
        messages.extend(history[-8:])
    messages.append(
        {
            "role": "user",
            "content": f"项目资料：\n\n{context}\n\n用户问题：{query}",
        }
    )
    return call_llm_messages(messages, temperature=0.2)


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def local_analyze_feedback(text: str) -> dict[str, str]:
    dimension_map = {
        "感知": ["识别", "漏检", "误检", "障碍物", "目标", "看不到"],
        "决策": ["犹豫", "变道", "规划", "决策", "超车", "让行"],
        "交互": ["提醒", "接管", "提示", "声音", "显示", "预期"],
        "硬件适配": ["算力", "传感器", "标定", "卡顿", "延迟", "摄像头"],
        "场景边界": ["降级", "退出", "无法开启", "ODD", "园区", "环岛", "掉头"],
    }
    dimensions = [
        dimension
        for dimension, keywords in dimension_map.items()
        if any(keyword in text for keyword in keywords)
    ] or ["待进一步定位"]

    if any(word in text for word in ["事故", "碰撞", "撞向", "紧急", "失控"]):
        severity = "S"
    elif any(word in text for word in ["降级", "退出", "误识别", "漏检", "无法开启"]):
        severity = "A"
    elif any(word in text for word in ["犹豫", "提醒较晚", "体验差", "不顺畅"]):
        severity = "B"
    else:
        severity = "C"

    return {
        "scenario": "待补充场景",
        "observation": text.strip(),
        "severity": severity,
        "dimensions": dimensions,
        "suggested_owner": "产品运营牵头，联合研发与测试定位",
        "short_term_action": "补充测试记录、视频或日志，确认触发条件。",
        "verification": "同场景复测，观察问题是否可稳定复现。",
        "risk": "待进一步判断安全影响和发生频率。",
    }


def analyze_feedback(text: str) -> dict[str, Any]:
    context = rag_context(text)
    system_prompt = (
        "你是智驾产品体验分析助手。请把用户反馈转成结构化 JSON。"
        "只返回 JSON，不要输出解释。字段为 scenario、observation、severity、"
        "dimensions、suggested_owner、short_term_action、verification、risk。"
        "scenario 只能是 城市道路、高速道路、变道、跟车、避障、泊车、园区道路、掉头、环岛 中的一个。"
        "优先选择更具体的动作场景，例如包含变道就选变道，包含泊车就选泊车，包含掉头就选掉头。"
        "severity 只能是 S、A、B、C。dimensions 只能是感知、决策、交互、"
        "硬件适配、场景边界中的一个或多个。无法确认的字段填“待补充证据”。"
    )
    llm_text = call_llm_request(
        system_prompt,
        f"项目资料：\n\n{context}\n\n用户反馈：{text}",
        temperature=0.1,
        json_mode=True,
    )
    parsed = parse_json_response(llm_text) if llm_text else None
    if parsed:
        result = {
            "analysis": parsed,
            "mode": "llm",
            "sources": rag_sources(text),
        }
    else:
        result = {
            "analysis": local_analyze_feedback(text),
            "mode": "local",
            "sources": rag_sources(text),
        }
    saved = save_case(
        {
            "input": text,
            "analysis": result["analysis"],
            "mode": result["mode"],
            "sources": result["sources"],
        }
    )
    result["case_id"] = saved["id"]
    return result


def run_evaluation() -> dict[str, Any]:
    cases = load_eval_cases()
    results: list[dict[str, Any]] = []
    for case in cases:
        result = analyze_feedback(case["input"])
        actual = result.get("analysis", {})
        expected = case.get("expected", {})
        expected_dims = set(expected.get("dimensions", []))
        actual_dims = set(actual.get("dimensions", []))
        if not isinstance(actual_dims, set):
            actual_dims = set(actual_dims)
        dimension_recall = (
            len(expected_dims & actual_dims) / len(expected_dims)
            if expected_dims
            else 1.0
        )
        scenario_correct = actual.get("scenario") == expected.get("scenario")
        severity_correct = actual.get("severity") == expected.get("severity")
        case_score = (
            (1.0 if scenario_correct else 0.0)
            + (1.0 if severity_correct else 0.0)
            + dimension_recall
        ) / 3.0
        results.append(
            {
                "id": case["id"],
                "scenario_correct": scenario_correct,
                "severity_correct": severity_correct,
                "dimension_recall": round(dimension_recall, 3),
                "score": round(case_score, 3),
                "actual": actual,
                "expected": expected,
                "mode": result["mode"],
            }
        )
    avg_score = sum(item["score"] for item in results) / len(results) if results else 0.0
    return {
        "avg_score": round(avg_score, 3),
        "total": len(results),
        "results": results,
    }


def system_metrics() -> dict[str, Any]:
    logs = load_model_logs(200)
    latencies = [float(item.get("latency_ms", 0)) for item in logs if item.get("latency_ms")]
    return {
        "rag_chunks": len(RAG.chunks),
        "documents": len(DOC_FILES),
        "cases": len(load_cases()),
        "eval_cases": len(load_eval_cases()),
        "model_calls": len(logs),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "mode": "llm" if llm_endpoint() else "local",
        "vector_rag": bool(VECTOR_RAG and VECTOR_RAG.available),
    }


def generate_report() -> dict[str, Any]:
    cases = load_cases()
    metrics = system_metrics()
    report_lines = [
        "# 智驾产品体验运营 Agent 运行报告",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        f"- 运行模式：{metrics['mode']}",
        f"- RAG 文档块：{metrics['rag_chunks']}",
        f"- 案例数量：{metrics['cases']}",
        f"- 模型调用次数：{metrics['model_calls']}",
        f"- 平均模型耗时：{metrics['avg_latency_ms']} ms",
        "",
        "## 最近案例",
        "",
    ]
    if cases:
        for case in cases[:5]:
            analysis = case.get("analysis", {})
            report_lines.append(f"- [{case.get('id')}] {case.get('input', '')}")
            report_lines.append(
                f"  - 场景：{analysis.get('scenario', '')} | 严重度：{analysis.get('severity', '')} | 维度：{analysis.get('dimensions', [])}"
            )
    else:
        report_lines.append("- 暂无案例")
    report = "\n".join(report_lines)
    return {"report": report, "metrics": metrics, "case_count": len(cases)}


def answer(query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    context = rag_context(query)
    sources = rag_sources(query)
    llm_answer = call_llm(query, context, history)
    if llm_answer:
        return {
            "reply": llm_answer,
            "mode": "llm",
            "sources": sources,
        }
    return {
        "reply": local_answer(query),
        "mode": "local",
        "sources": sources,
    }


class AgentHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "mode": "llm" if llm_endpoint() else "local",
                    "rag_chunks": len(RAG.chunks),
                    "documents": len(DOC_FILES),
                    "vector_rag": bool(VECTOR_RAG and VECTOR_RAG.available),
                }
            )
            return
        if self.path == "/api/cases":
            self.send_json({"cases": load_cases()})
            return
        if self.path == "/api/metrics":
            self.send_json(system_metrics())
            return
        if self.path == "/api/report":
            self.send_json(generate_report())
            return
        if self.path == "/api/logs":
            self.send_json({"logs": load_model_logs(100)})
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            if self.path == "/api/chat":
                query = str(payload.get("message", "")).strip()
                if not query:
                    self.send_json({"error": "message is required"}, status=400)
                    return
                history = payload.get("history") or []
                if not isinstance(history, list):
                    history = []
                self.send_json(answer(query, history=history[-8:]))
                return
            if self.path == "/api/analyze":
                text = str(payload.get("text", "")).strip()
                if not text:
                    self.send_json({"error": "text is required"}, status=400)
                    return
                self.send_json(analyze_feedback(text))
                return
            if self.path == "/api/eval":
                self.send_json(run_evaluation())
                return
            if self.path == "/api/rag/search":
                query = str(payload.get("message", "")).strip()
                if not query:
                    self.send_json({"error": "message is required"}, status=400)
                    return
                top_k = int(payload.get("top_k", 6))
                self.send_json(
                    {
                        "query": query,
                        "results": rag_search(query, top_k=top_k),
                    }
                )
                return
            self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global VECTOR_RAG
    load_dotenv()
    VECTOR_RAG = VectorRAG(ROOT, DOC_FILES)
    port = int(os.getenv("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AgentHandler)
    mode = "llm" if llm_endpoint() else "local"
    print(f"智驾运营 Agent running at http://127.0.0.1:{port}", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"vector_rag: {'available' if VECTOR_RAG.available else 'fallback'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"server error: {exc}", file=sys.stderr)
        raise
