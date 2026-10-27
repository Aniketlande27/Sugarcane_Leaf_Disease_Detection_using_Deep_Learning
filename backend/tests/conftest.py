"""Test setup: point the app at a throw-away database/upload/model folder BEFORE it is imported,
so running the tests can never touch your real app.db, uploads or model files."""
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

TMP = Path(tempfile.mkdtemp(prefix="canescan_test_"))
(TMP / "model").mkdir()
(TMP / "uploads").mkdir()
# Class names are test data only; there is deliberately NO model file here.
(TMP / "model" / "class_names.json").write_text(json.dumps(["Healthy", "Mosaic", "Red_Rot", "Rust"]))

os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(TMP / "uploads")
os.environ["MODEL_DIR"] = str(TMP / "model")
os.environ["LOAD_MODEL_ON_STARTUP"] = "true"
