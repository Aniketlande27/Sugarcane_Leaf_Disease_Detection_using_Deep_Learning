"""Central configuration. Every value can be overridden with an environment variable.

All paths are derived from this file's location, never from the terminal's current directory,
so `uvicorn main:app` works no matter where you start it from.
"""
import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str) -> Optional[Path]:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- model files
MODEL_DIR = _env_path("MODEL_DIR") or (BASE_DIR / "model")
# If MODEL_PATH is not set, predictor.py looks inside MODEL_DIR (see resolve_model_path()).
MODEL_PATH_OVERRIDE = _env_path("MODEL_PATH")
PREFERRED_MODEL_NAME = "sugarcane_disease_efficientnetB0.keras"     # final model saved by the training script
CLASS_NAMES_PATH = _env_path("CLASS_NAMES_PATH") or (MODEL_DIR / "class_names.json")
FALLBACK_IMG_SIZE = (224, 224)      # only used if the model's own input size cannot be read

# Load the model while the server starts (True), or on the first prediction request (False).
LOAD_MODEL_ON_STARTUP = _env_bool("LOAD_MODEL_ON_STARTUP", True)

# ---------------------------------------------------------------- storage
UPLOAD_DIR = _env_path("UPLOAD_DIR") or (BASE_DIR / "uploads")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}")
DISEASE_INFO_PATH = BASE_DIR / "disease_info.json"


def _find_frontend_dir() -> Path:
    override = _env_path("FRONTEND_DIR")
    if override:
        return override
    # Works whether the folder is called "frontend" or "Frontend", and on case-sensitive systems (Linux/WSL).
    for name in ("Frontend", "frontend"):
        candidate = BASE_DIR.parent / name
        if candidate.is_dir():
            return candidate
    return BASE_DIR.parent / "frontend"


FRONTEND_DIR = _find_frontend_dir()

# ---------------------------------------------------------------- behaviour
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MIN_IMAGE_SIDE = 32                                   # pixels; smaller images are rejected
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
