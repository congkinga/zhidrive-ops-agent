from fastapi.testclient import TestClient

from fastapi_app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["service"] == "zhidrive-ops-fastapi"
    assert data["langgraph"] is True


def test_cases_endpoint() -> None:
    response = client.get("/api/v1/cases")
    assert response.status_code == 200
    assert isinstance(response.json()["cases"], list)


def test_hybrid_retrieval() -> None:
    response = client.post(
        "/api/rag/hybrid",
        json={"message": "无保护左转 NOA 降级如何归因", "top_k": 3},
    )
    assert response.status_code == 200
    results = response.json().get("results", [])
    assert len(results) <= 3
    assert all("source" in item for item in results)


def test_ops_growth_endpoints() -> None:
    overview = client.get("/api/ops/overview")
    assert overview.status_code == 200
    assert "feedback_processed" in overview.json()

    funnel = client.get("/api/ops/funnel")
    assert funnel.status_code == 200
    assert "funnel" in funnel.json()

    segments = client.get("/api/ops/segments")
    assert segments.status_code == 200
    assert "segments" in segments.json()

    activities = client.get("/api/ops/activities")
    assert activities.status_code == 200
    assert "activities" in activities.json()
