"""API tests. Need: pip install -r requirements-dev.txt
Tests marked (stub) use a stand-in predictor ONLY to exercise upload validation and database code;
they say nothing about the real model - see test_real_model_inference_uses_real_classes for that."""
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient    # noqa: E402
from PIL import Image                         # noqa: E402

import config                                 # noqa: E402
import main                                   # noqa: E402
import predictor as model_service             # noqa: E402
from predictor import InferenceError          # noqa: E402


class StubPredictor:
    class_names = ["Healthy", "Mosaic", "Red_Rot", "Rust"]

    def predict(self, image):
        return {"label": "Rust", "confidence": 0.93,
                "all_probs": {"Healthy": 0.02, "Mosaic": 0.02, "Red_Rot": 0.03, "Rust": 0.93}}


class FailingPredictor:
    def predict(self, image):
        raise InferenceError("boom")


def make_image(fmt="PNG", size=(120, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 160, 70)).save(buf, fmt)
    return buf.getvalue()


def upload(client, data, name="leaf.png", ctype="image/png"):
    return client.post("/api/predict", files={"file": (name, data, ctype)})


@pytest.fixture()
def client():
    model_service._reset_for_tests()
    with TestClient(main.app) as c:      # runs the real startup (no model file exists in the test folder)
        yield c
    model_service._reset_for_tests()


@pytest.fixture()
def ready_client(client):
    model_service._install_for_tests(StubPredictor())
    return client


# ------------------------------------------------ model unavailable: API stays up, no fake predictions
def test_health_is_alive_without_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database_ok": True, "model_ready": False}


def test_ready_is_503_with_safe_diagnosis(client):
    r = client.get("/api/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and body["code"] == "model_file_missing"
    assert str(config.MODEL_DIR) not in r.text          # no filesystem paths leak to users


def test_predict_returns_503_not_a_fake_prediction(client):
    r = upload(client, make_image())
    assert r.status_code == 503
    body = r.json()
    assert "label" not in body and "confidence" not in body
    assert body["code"] == "model_file_missing"


def test_non_inference_endpoints_work_without_model(client):
    assert client.get("/api/history").status_code == 200
    assert client.get("/api/stats").status_code == 200
    diseases = client.get("/api/diseases").json()
    assert {d["name"] for d in diseases} >= {"Healthy", "Mosaic", "Red_Rot", "Rust"}


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CaneScan" in r.text


# ------------------------------------------------ (stub) upload validation and database
def test_stub_valid_upload_is_saved_and_returned(ready_client):
    before = ready_client.get("/api/history").json()["total"]
    r = upload(ready_client, make_image())
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Rust" and body["disease"]["display_name"] == "Rust"
    assert {"id", "confidence", "is_uncertain", "threshold", "top3", "image_url", "disease"} <= set(body)
    assert ready_client.get("/api/history").json()["total"] == before + 1
    assert ready_client.get(body["image_url"]).status_code == 200


def test_stub_corrupt_image_rejected(ready_client):
    r = upload(ready_client, make_image()[:60])
    assert r.status_code == 400 and r.json()["code"] == "invalid_image"


def test_stub_non_image_rejected(ready_client):
    r = upload(ready_client, b"just some text", "x.png")
    assert r.status_code == 400 and r.json()["code"] == "invalid_image"


def test_stub_unsupported_format_rejected(ready_client):
    r = upload(ready_client, make_image("GIF"), "x.gif", "image/gif")
    assert r.status_code == 400 and r.json()["code"] == "unsupported_format"


def test_stub_tiny_image_rejected(ready_client):
    r = upload(ready_client, make_image(size=(8, 8)))
    assert r.status_code == 400 and r.json()["code"] == "image_too_small"


def test_stub_oversized_upload_rejected(ready_client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    r = upload(ready_client, b"0" * (1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["code"] == "file_too_large"


def test_stub_empty_file_rejected(ready_client):
    assert upload(ready_client, b"").status_code == 400


def test_missing_file_gives_consistent_422(ready_client):
    r = ready_client.post("/api/predict")
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_inference_failure_is_an_error_not_a_prediction(client):
    model_service._install_for_tests(FailingPredictor())
    before = client.get("/api/history").json()["total"]
    r = upload(client, make_image())
    assert r.status_code == 500 and r.json()["code"] == "inference_failed" and "label" not in r.json()
    assert client.get("/api/history").json()["total"] == before     # nothing stored for a failed run


def test_feedback_flow(ready_client):
    pid = upload(ready_client, make_image()).json()["id"]
    assert ready_client.post(f"/api/predictions/{pid}/feedback", json={"correct": True}).status_code == 200
    assert ready_client.post(f"/api/predictions/{pid}/feedback", json={"correct": False}).status_code == 400
    assert ready_client.post(f"/api/predictions/{pid}/feedback", json={"correct": False, "correct_label": "Nope"}).status_code == 400
    assert ready_client.post(f"/api/predictions/{pid}/feedback", json={"correct": False, "correct_label": "Mosaic"}).status_code == 200
    assert ready_client.post("/api/predictions/999999/feedback", json={"correct": True}).status_code == 404


def test_existing_records_survive_a_restart():
    model_service._reset_for_tests()
    with TestClient(main.app) as first:
        model_service._install_for_tests(StubPredictor())
        upload(first, make_image())
        total = first.get("/api/history").json()["total"]
    model_service._reset_for_tests()
    with TestClient(main.app) as second:          # startup runs again (create_all + seed)
        assert second.get("/api/history").json()["total"] == total
