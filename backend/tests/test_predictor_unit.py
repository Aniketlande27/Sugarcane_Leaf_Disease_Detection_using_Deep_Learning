"""Unit tests for model loading/diagnostics. They do NOT need FastAPI, and (except the last one) not TensorFlow."""
import json

import numpy as np
import pytest
from PIL import Image

import config
import predictor
from predictor import ModelUnavailable


# ---------------------------------------------------------------- error classification
def test_windows_code_integrity_error_is_recognised():
    exc = ImportError("DLL load failed while importing _dtypes: An Application Control policy has blocked this file.")
    code, message = predictor.classify_import_error(exc)
    assert code == "tensorflow_blocked_by_policy"
    assert "Windows security" in message


def test_missing_tensorflow_is_recognised():
    code, _ = predictor.classify_import_error(ModuleNotFoundError("No module named 'tensorflow'"))
    assert code == "tensorflow_not_installed"


def test_other_dll_error_is_recognised():
    code, _ = predictor.classify_import_error(ImportError("DLL load failed while importing something: The specified module could not be found."))
    assert code == "tensorflow_import_failed"


# ---------------------------------------------------------------- class names
def test_class_names_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", tmp_path / "nope.json")
    with pytest.raises(ModelUnavailable) as e:
        predictor.load_class_names()
    assert e.value.code == "class_names_missing"


@pytest.mark.parametrize("content", ["not json", "{}", '["only_one"]', '["a", "a"]', '["a", ""]'])
def test_class_names_invalid(monkeypatch, tmp_path, content):
    p = tmp_path / "c.json"
    p.write_text(content)
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", p)
    with pytest.raises(ModelUnavailable) as e:
        predictor.load_class_names()
    assert e.value.code == "class_names_invalid"


def test_class_names_valid_keeps_order(monkeypatch, tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["B", "A", "C"]))
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", p)
    assert predictor.load_class_names() == ["B", "A", "C"]


# ---------------------------------------------------------------- model file discovery
def _point_at(monkeypatch, folder):
    monkeypatch.setattr(config, "MODEL_DIR", folder)
    monkeypatch.setattr(config, "MODEL_PATH_OVERRIDE", None)


def test_no_model_file(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    with pytest.raises(ModelUnavailable) as e:
        predictor.resolve_model_path()
    assert e.value.code == "model_file_missing"


def test_single_model_file_is_found(monkeypatch, tmp_path):
    (tmp_path / "whatever_name.keras").write_bytes(b"x")
    _point_at(monkeypatch, tmp_path)
    assert predictor.resolve_model_path().name == "whatever_name.keras"


def test_preferred_final_model_wins_over_stage_checkpoints(monkeypatch, tmp_path):
    for name in ("best_model_stage1.keras", "best_model_stage2.keras", config.PREFERRED_MODEL_NAME):
        (tmp_path / name).write_bytes(b"x")
    _point_at(monkeypatch, tmp_path)
    assert predictor.resolve_model_path().name == config.PREFERRED_MODEL_NAME


def test_several_unknown_models_is_ambiguous(monkeypatch, tmp_path):
    (tmp_path / "a.keras").write_bytes(b"x")
    (tmp_path / "b.keras").write_bytes(b"x")
    _point_at(monkeypatch, tmp_path)
    with pytest.raises(ModelUnavailable) as e:
        predictor.resolve_model_path()
    assert e.value.code == "model_ambiguous"


# ---------------------------------------------------------------- state handling: failures are reported, never faked
def test_initialize_reports_missing_model_and_get_predictor_raises(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    predictor._reset_for_tests()
    status = predictor.initialize_model(["A", "B"])
    assert status["ready"] is False and status["code"] == "model_file_missing"
    with pytest.raises(ModelUnavailable):
        predictor.get_predictor()


def test_initialize_reports_tensorflow_problem(monkeypatch, tmp_path):
    try:
        import tensorflow  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("TensorFlow imports fine here; this test needs an environment where it does not.")
    (tmp_path / "dummy.keras").write_bytes(b"not a real model")
    _point_at(monkeypatch, tmp_path)
    predictor._reset_for_tests()
    status = predictor.initialize_model(["A", "B"])
    assert status["ready"] is False
    assert status["code"] in {"tensorflow_not_installed", "tensorflow_blocked_by_policy", "tensorflow_import_failed"}
    with pytest.raises(ModelUnavailable):
        predictor.get_predictor()


def test_invalid_model_file_is_reported(monkeypatch, tmp_path):
    try:
        import tensorflow  # noqa: F401
    except Exception:
        pytest.skip("Needs a working TensorFlow.")
    (tmp_path / "broken.keras").write_bytes(b"this is not a keras model")
    _point_at(monkeypatch, tmp_path)
    predictor._reset_for_tests()
    assert predictor.initialize_model(["A", "B"])["code"] == "model_load_failed"


# ---------------------------------------------------------------- REAL inference (needs your model + working TensorFlow)
def test_real_model_inference_uses_real_classes(monkeypatch):
    real_dir = config.BASE_DIR / "model"
    names_file = real_dir / "class_names.json"
    has_model = any(p.suffix.lower() in {".keras", ".h5"} for p in real_dir.glob("*"))
    if not (names_file.is_file() and has_model):
        pytest.skip("Real model files are not in backend/model.")
    try:
        import tensorflow  # noqa: F401
    except Exception as exc:
        pytest.skip(f"TensorFlow cannot be imported here: {type(exc).__name__}")

    monkeypatch.setattr(config, "MODEL_DIR", real_dir)
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", names_file)
    monkeypatch.setattr(config, "MODEL_PATH_OVERRIDE", None)
    predictor._reset_for_tests()
    names = json.loads(names_file.read_text())
    status = predictor.initialize_model(names)
    assert status["ready"], status

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    result = predictor.get_predictor().predict(img)
    assert result["label"] in names
    assert list(result["all_probs"]) == names                      # class order preserved
    assert abs(sum(result["all_probs"].values()) - 1.0) < 1e-3     # genuine softmax output
    predictor._reset_for_tests()
