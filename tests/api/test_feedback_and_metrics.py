from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import feedback as feedback_api
from app.api import metrics as metrics_api
from app.api import routes as qa_routes
from app.core.config import get_settings
from app.services.telemetry import TelemetryLogger
from app.services.feedback import FeedbackRepository
from app.main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    telemetry_path = tmp_path / "telemetry.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("TELEMETRY_LOG_PATH", str(telemetry_path))
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(feedback_path))

    get_settings.cache_clear()
    qa_routes._get_telemetry_logger.cache_clear()
    feedback_api._get_feedback_repo.cache_clear()
    feedback_api._get_telemetry_logger.cache_clear()
    metrics_api._get_feedback_repo.cache_clear()
    metrics_api._get_telemetry_logger.cache_clear()

    return TestClient(app)


def test_feedback_requires_known_interaction(client: TestClient) -> None:
    payload = {
        "interaction_id": uuid4().hex,
        "rating": "helpful",
        "comment": None,
        "correction": None,
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 404


def test_feedback_submission_and_metrics(client: TestClient) -> None:
    settings = get_settings()
    telemetry_logger = TelemetryLogger(path=Path(settings.telemetry_log_path))
    feedback_repo = FeedbackRepository(path=Path(settings.feedback_log_path))

    interaction_id = uuid4().hex
    telemetry_logger.log_answer(
        interaction_id=interaction_id,
        question="Test question",
        normalized_question="Test question",
        language="fr",
        answer="Test answer",
        confidence=0.8,
        abstain=False,
        sources=["doc-test"],
    )

    payload = {
        "interaction_id": interaction_id,
        "rating": "helpful",
        "comment": "Super",
        "correction": None,
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 202
    body = response.json()
    assert body["interaction_id"] == interaction_id
    assert body["outcome"] == pytest.approx(1.0)

    metrics_response = client.get("/api/v1/metrics/calibration")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["total_interactions"] == 1
    assert metrics["feedback_count"] == 1
    assert metrics["ece"] == pytest.approx(metrics["ece"])
    assert metrics["brier"] == pytest.approx(metrics["brier"])
    assert metrics["bins"]

    # Ensure the feedback was persisted once
    records = list(feedback_repo.iter_records())
    assert len(records) == 1
    assert records[0].interaction_id == interaction_id