"""Model loading and inference.

Design rules
------------
* TensorFlow is imported ONLY inside `_build_predictor()`, never at module import time. The API,
  database and frontend therefore keep working even if TensorFlow cannot be loaded.
* `initialize_model()` is the single, centralized place that loads the model. It never raises: it records a
  status (ready / not ready, plus a machine-readable code and a safe message) that /api/ready reports.
* If anything fails there is NO fallback prediction. `get_predictor()` raises `ModelUnavailable`, which the API
  turns into an HTTP 503. Full technical details (file paths, tracebacks) go to the server log only.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

import config

log = logging.getLogger("canescan.model")


# ============================================================================ errors
class ModelUnavailable(Exception):
    """The model (or something it needs) is not usable. `message`/`hint` are safe to show to users."""

    def __init__(self, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


class InferenceError(Exception):
    """The model was loaded but failed on one particular image."""


HINTS = {
    "tensorflow_blocked_by_policy": (
        "Windows Application Control (Code Integrity) is blocking a TensorFlow file. This cannot be fixed "
        "in the application code. Ask the device administrator to approve it, or run the backend in WSL2/Linux, "
        "Docker or a cloud server. See the README section 'TensorFlow blocked by Windows'."
    ),
    "tensorflow_not_installed": "Install the ML dependencies: pip install -r requirements-ml.txt",
    "tensorflow_import_failed": "Run `python diagnose_tf.py` in the backend folder for details.",
    "model_file_missing": "Copy the trained .keras model into the backend/model folder (or set MODEL_PATH).",
    "model_ambiguous": "Several model files were found. Set the MODEL_PATH environment variable to the one to use.",
    "class_names_missing": "Copy class_names.json (saved by the training script) into the backend/model folder.",
    "class_names_invalid": "class_names.json must be a JSON list of at least two unique class-name strings.",
    "model_load_failed": (
        "The model file could not be read. Make sure the TensorFlow/Keras version matches the one used for training."
    ),
    "model_input_unsupported": "The model must take a single 4-D RGB image input (height, width, 3).",
    "class_count_mismatch": "The model's output size does not match class_names.json. Use files from the same training run.",
}


# ============================================================================ pure helpers (no TensorFlow)
def classify_import_error(exc: BaseException) -> tuple[str, str]:
    """Turn a failed `import tensorflow` into (code, safe message)."""
    text = str(exc).lower()
    if isinstance(exc, ModuleNotFoundError) and "tensorflow" in text:
        return "tensorflow_not_installed", "TensorFlow is not installed in this Python environment."
    if "application control" in text or "blocked this file" in text or "code integrity" in text:
        return (
            "tensorflow_blocked_by_policy",
            "Windows security (Application Control policy) blocked a TensorFlow file, so the model cannot be loaded.",
        )
    if "dll load failed" in text:
        return "tensorflow_import_failed", "TensorFlow could not load a required system library (DLL)."
    return "tensorflow_import_failed", "TensorFlow could not be imported."


def load_class_names() -> list[str]:
    """Read class_names.json. Does not need TensorFlow."""
    path = config.CLASS_NAMES_PATH
    if not path.is_file():
        log.error("class_names.json not found at %s", path)
        raise ModelUnavailable("class_names_missing", "The class names file is missing.", HINTS["class_names_missing"])
    try:
        with open(path, encoding="utf-8") as f:
            names = json.load(f)
    except (OSError, ValueError) as exc:
        log.error("Could not read class names from %s: %s", path, exc)
        raise ModelUnavailable("class_names_invalid", "The class names file could not be read.", HINTS["class_names_invalid"])
    if (
        not isinstance(names, list)
        or len(names) < 2
        or not all(isinstance(n, str) and n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ModelUnavailable("class_names_invalid", "The class names file has an invalid format.", HINTS["class_names_invalid"])
    return names


def resolve_model_path() -> Path:
    """Find the trained model without assuming a file name."""
    if config.MODEL_PATH_OVERRIDE is not None:
        if config.MODEL_PATH_OVERRIDE.is_file():
            return config.MODEL_PATH_OVERRIDE
        log.error("MODEL_PATH points to a file that does not exist: %s", config.MODEL_PATH_OVERRIDE)
        raise ModelUnavailable("model_file_missing", "The configured model file does not exist.", HINTS["model_file_missing"])

    preferred = config.MODEL_DIR / config.PREFERRED_MODEL_NAME
    if preferred.is_file():
        return preferred

    candidates = sorted(p for p in config.MODEL_DIR.glob("*") if p.suffix.lower() in {".keras", ".h5"})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        log.error("No .keras/.h5 model found in %s", config.MODEL_DIR)
        raise ModelUnavailable("model_file_missing", "No trained model file was found.", HINTS["model_file_missing"])
    log.error("Several model files found in %s: %s", config.MODEL_DIR, [p.name for p in candidates])
    raise ModelUnavailable("model_ambiguous", "More than one model file was found.", HINTS["model_ambiguous"])


# ============================================================================ the predictor
class Predictor:
    """Wraps a loaded Keras model. Preprocessing matches training: float32 pixels in 0-255 passed through
    EfficientNet's `preprocess_input` (applied outside the model, exactly like the training pipeline)."""

    def __init__(self, model, class_names: list[str], img_size: tuple[int, int], preprocess):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size                      # (height, width) read from the model itself
        self._preprocess = preprocess
        self._lock = threading.Lock()                 # one inference at a time per process

    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        x = self._preprocess(x.copy())
        with self._lock:
            p_plain = self.model.predict(x, verbose=0)
            p_flip = self.model.predict(x[:, :, ::-1, :], verbose=0)    # horizontal-flip TTA, as in evaluation
        return ((p_plain + p_flip) / 2.0)[0]

    def predict(self, image: Image.Image) -> dict:
        try:
            h, w = self.img_size
            image = ImageOps.exif_transpose(image).convert("RGB").resize((w, h), Image.BILINEAR)
            x = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
            probs = self._predict_array(x)
            if probs.shape[0] != len(self.class_names) or not np.all(np.isfinite(probs)):
                raise ValueError("model returned an invalid probability vector")
            idx = int(np.argmax(probs))
            return {
                "label": self.class_names[idx],
                "confidence": float(probs[idx]),
                "all_probs": {c: float(p) for c, p in zip(self.class_names, probs)},
            }
        except Exception as exc:      # noqa: BLE001 - logged in full, reported generically
            log.exception("Inference failed")
            raise InferenceError(str(exc)) from exc


def _build_predictor(class_names: list[str]) -> Predictor:
    """The ONLY function that imports TensorFlow and loads the model."""
    model_path = resolve_model_path()

    try:
        from tensorflow import keras                                                   # noqa: WPS433
        from tensorflow.keras.applications.efficientnet import preprocess_input        # noqa: WPS433
    except Exception as exc:       # ImportError, ModuleNotFoundError, OSError ... never hide it
        code, message = classify_import_error(exc)
        log.exception("TensorFlow import failed (%s)", code)
        raise ModelUnavailable(code, message, HINTS.get(code)) from exc

    try:
        model = keras.models.load_model(model_path, compile=False)    # compile=False: inference only, no optimizer needed
    except Exception as exc:       # noqa: BLE001
        log.exception("Failed to load model %s", model_path)
        raise ModelUnavailable("model_load_failed", "The model file could not be loaded.", HINTS["model_load_failed"]) from exc

    shape = model.input_shape
    if not (isinstance(shape, tuple) and len(shape) == 4 and shape[-1] == 3):
        log.error("Unsupported model input shape: %s", shape)
        raise ModelUnavailable("model_input_unsupported", "The model has an unsupported input shape.", HINTS["model_input_unsupported"])
    height, width = shape[1], shape[2]
    if not height or not width:
        height, width = config.FALLBACK_IMG_SIZE

    out_shape = model.output_shape
    n_out = out_shape[-1] if isinstance(out_shape, tuple) else None
    if n_out != len(class_names):
        log.error("Model has %s outputs but class_names.json has %s entries", n_out, len(class_names))
        raise ModelUnavailable("class_count_mismatch", "The model and class names file do not match.", HINTS["class_count_mismatch"])

    predictor = Predictor(model, class_names, (int(height), int(width)), preprocess_input)
    try:                                                    # warm-up proves inference really works
        predictor._predict_array(np.zeros((1, int(height), int(width), 3), dtype=np.float32))
    except Exception as exc:       # noqa: BLE001
        log.exception("Model warm-up failed")
        raise ModelUnavailable("model_load_failed", "The model loaded but could not run a test prediction.", HINTS["model_load_failed"]) from exc

    log.info("Model ready: %s | input %sx%s | %d classes", model_path.name, height, width, len(class_names))
    return predictor


# ============================================================================ shared state
_state_lock = threading.Lock()
_predictor: Optional[Predictor] = None
_status: dict = {"ready": False, "code": "not_initialized", "message": "The model has not been loaded yet.", "hint": None}


def initialize_model(class_names: Optional[list[str]] = None) -> dict:
    """Load the model once. Never raises; returns (and stores) the resulting status."""
    global _predictor, _status
    with _state_lock:
        try:
            names = class_names if class_names is not None else load_class_names()
            _predictor = _build_predictor(names)
            _status = {"ready": True, "code": "ready", "message": "Model is ready.", "hint": None}
        except ModelUnavailable as exc:
            _predictor = None
            _status = {"ready": False, "code": exc.code, "message": exc.message, "hint": exc.hint}
        except Exception:      # noqa: BLE001 - anything unexpected is still reported, never swallowed
            log.exception("Unexpected error while loading the model")
            _predictor = None
            _status = {
                "ready": False, "code": "unexpected_startup_error",
                "message": "The model could not be started because of an unexpected error. See the server log.",
                "hint": None,
            }
        return dict(_status)


def get_status() -> dict:
    return dict(_status)


def get_predictor() -> Predictor:
    """Return the ready predictor or raise ModelUnavailable. Loads lazily if startup loading is switched off."""
    if _predictor is None and _status["code"] == "not_initialized":
        initialize_model()
    if _predictor is None:
        s = _status
        raise ModelUnavailable(s["code"], s["message"], s["hint"])
    return _predictor


# ---- test helpers (used only by the test-suite) ------------------------------------------------
def _install_for_tests(predictor: object) -> None:
    global _predictor, _status
    _predictor = predictor      # type: ignore[assignment]
    _status = {"ready": True, "code": "ready", "message": "Model is ready.", "hint": None}


def _reset_for_tests() -> None:
    global _predictor, _status
    _predictor = None
    _status = {"ready": False, "code": "not_initialized", "message": "The model has not been loaded yet.", "hint": None}
