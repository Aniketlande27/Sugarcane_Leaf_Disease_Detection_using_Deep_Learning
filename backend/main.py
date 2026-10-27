"""Sugarcane disease detection API.

Run from the backend folder:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000  (frontend)  or  http://127.0.0.1:8000/docs  (API docs).

The API starts even if TensorFlow or the model cannot be loaded. In that case:
  * /api/health  -> 200 (the API is alive, "model_ready": false)
  * /api/ready   -> 503 with a safe explanation
  * /api/predict -> 503 with a safe explanation (never a fake prediction)
  * history, disease guide, stats and the frontend keep working.
"""
import io
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlalchemy.orm import Session

import config
import predictor as model_service
from database import Base, engine, SessionLocal, get_db
from disease_data import seed_diseases
from models import Prediction, Disease
from predictor import ModelUnavailable, InferenceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("canescan.api")


# ============================================================================ errors with a consistent shape
class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, hint: Optional[str] = None):
        self.status_code, self.code, self.message, self.hint = status_code, code, message, hint


def _error_body(code: str, message: str, hint: Optional[str] = None) -> dict:
    # "detail" is kept as a plain string because the frontend reads it.
    body = {"detail": message, "code": code}
    if hint:
        body["hint"] = hint
    return body


# ============================================================================ startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)       # creates missing tables only; never drops or alters existing data

    class_names: list[str] = []
    try:
        class_names = model_service.load_class_names()
    except ModelUnavailable as exc:
        log.error("Class names unavailable (%s): %s", exc.code, exc.message)
    app.state.class_names = class_names

    if class_names:
        with SessionLocal() as db:
            missing = seed_diseases(db, class_names)
        if missing:
            log.info("No specific advice in disease_info.json for %s (generic text used).", missing)

    if config.LOAD_MODEL_ON_STARTUP:
        status = model_service.initialize_model(class_names or None)
        if status["ready"]:
            log.info("Model ready with classes: %s", class_names)
        else:
            log.error("Model NOT ready (%s): %s | The API will run without prediction.", status["code"], status["message"])
    else:
        log.info("LOAD_MODEL_ON_STARTUP is off; the model will be loaded on the first prediction request.")
    yield


app = FastAPI(title="Sugarcane Disease Detection", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=_error_body(exc.code, exc.message, exc.hint))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=_error_body("invalid_request", "The request was missing or had invalid data."))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)      # full trace only in the server log
    return JSONResponse(status_code=500, content=_error_body("internal_error", "Something went wrong on the server."))


# ============================================================================ helpers
def disease_to_dict(d: Disease) -> dict:
    return {
        "name": d.name,
        "display_name": d.display_name,
        "severity": d.severity,
        "description": d.description,
        "symptoms": d.symptoms,
        "treatment": d.treatment,
        "prevention": d.prevention,
    }


def prediction_to_dict(p: Prediction) -> dict:
    return {
        "id": p.id,
        "label": p.predicted_label,
        "confidence": p.confidence,
        "is_uncertain": p.is_uncertain,
        "image_url": f"/uploads/{p.image_file}",
        "feedback_correct": p.feedback_correct,
        "feedback_label": p.feedback_label,
        "created_at": p.created_at.isoformat(),
    }


def read_and_validate_image(file: UploadFile) -> tuple[bytes, Image.Image, str]:
    """Returns (raw bytes, decoded image, extension) or raises ApiError."""
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > limit:
        raise ApiError(413, "file_too_large", f"The image is larger than {config.MAX_UPLOAD_MB} MB. Use a smaller photo.")

    try:
        Image.open(io.BytesIO(data)).verify()              # detects truncated/corrupt files
        image = Image.open(io.BytesIO(data))               # verify() invalidates the object, so reopen
        fmt = image.format
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError, ValueError, Image.DecompressionBombError):
        raise ApiError(400, "invalid_image", "That file is not a readable image. Upload a JPG, PNG or WEBP photo.")

    if fmt not in config.ALLOWED_FORMATS:
        raise ApiError(400, "unsupported_format", "Unsupported image type. Upload a JPG, PNG or WEBP photo.")
    if min(image.size) < config.MIN_IMAGE_SIDE:
        raise ApiError(400, "image_too_small", f"The image is too small. Use a photo at least {config.MIN_IMAGE_SIDE}px wide and tall.")
    return data, image, {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]


# ============================================================================ health / readiness
@app.get("/api/health")
def health():
    """Liveness: is the API process up? Also reports whether the model and database are usable."""
    db_ok = True
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        log.exception("Database health check failed")
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database_ok": db_ok, "model_ready": model_service.get_status()["ready"]}


@app.get("/api/ready")
def ready():
    """Readiness: can the API actually make predictions right now?"""
    status = model_service.get_status()
    if status["ready"]:
        return {"ready": True, "code": "ready", "message": status["message"]}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "code": status["code"], "message": status["message"], "hint": status["hint"]},
    )


# ============================================================================ prediction
@app.post("/api/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        predictor = model_service.get_predictor()
    except ModelUnavailable as exc:
        raise ApiError(503, exc.code, f"Diagnosis is temporarily unavailable. {exc.message}", exc.hint)

    data, image, ext = read_and_validate_image(file)

    try:
        result = predictor.predict(image)
    except InferenceError:
        raise ApiError(500, "inference_failed", "The model could not analyse this image. Try a different photo.")

    filename = f"{uuid.uuid4().hex}{ext}"                  # never trust the client's file name
    (config.UPLOAD_DIR / filename).write_bytes(data)
    uncertain = result["confidence"] < config.CONFIDENCE_THRESHOLD

    record = Prediction(
        image_file=filename,
        predicted_label=result["label"],
        confidence=result["confidence"],
        all_probs=result["all_probs"],
        is_uncertain=uncertain,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    disease = db.query(Disease).filter(Disease.name == result["label"]).first()
    top3 = sorted(result["all_probs"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "id": record.id,
        "label": result["label"],
        "confidence": result["confidence"],
        "is_uncertain": uncertain,
        "threshold": config.CONFIDENCE_THRESHOLD,
        "top3": [{"label": k, "probability": v} for k, v in top3],
        "image_url": f"/uploads/{filename}",
        "disease": disease_to_dict(disease) if disease else None,
    }


# ============================================================================ data endpoints (work without the model)
@app.get("/api/history")
def history(limit: int = Query(12, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    total = db.query(func.count(Prediction.id)).scalar()
    rows = db.query(Prediction).order_by(Prediction.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [prediction_to_dict(r) for r in rows]}


class FeedbackIn(BaseModel):
    correct: bool
    correct_label: Optional[str] = None


@app.post("/api/predictions/{prediction_id}/feedback")
def feedback(prediction_id: int, body: FeedbackIn, db: Session = Depends(get_db)):
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise ApiError(404, "not_found", "Prediction not found.")
    if body.correct:
        record.feedback_label = record.predicted_label
    else:
        valid = {name for (name,) in db.query(Disease.name).all()}
        if not body.correct_label or body.correct_label not in valid:
            raise ApiError(400, "invalid_label", "Choose the correct disease from the list.")
        record.feedback_label = body.correct_label
    record.feedback_correct = body.correct
    db.commit()
    return {"ok": True}


@app.get("/api/diseases")
def diseases(db: Session = Depends(get_db)):
    rows = db.query(Disease).order_by(Disease.display_name).all()
    return [disease_to_dict(d) for d in rows]


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(func.count(Prediction.id)).scalar() or 0
    avg_conf = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
    uncertain = db.query(func.count(Prediction.id)).filter(Prediction.is_uncertain.is_(True)).scalar() or 0
    by_label = (
        db.query(Prediction.predicted_label, func.count(Prediction.id))
        .group_by(Prediction.predicted_label)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    reviewed = db.query(func.count(Prediction.id)).filter(Prediction.feedback_correct.is_not(None)).scalar() or 0
    confirmed = db.query(func.count(Prediction.id)).filter(Prediction.feedback_correct.is_(True)).scalar() or 0
    return {
        "total_scans": total,
        "average_confidence": float(avg_conf),
        "uncertain_scans": uncertain,
        "by_label": [{"label": l, "count": c} for l, c in by_label],
        "reviewed": reviewed,
        "confirmed_correct": confirmed,
    }


# ============================================================================ static files (must be mounted last)
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
