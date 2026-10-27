"""Loads disease_info.json and keeps the `diseases` table in sync with the model's class names."""
import json
import re

from sqlalchemy.orm import Session

from config import DISEASE_INFO_PATH
from models import Disease


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pretty(text: str) -> str:
    return re.sub(r"[_\-]+", " ", text).strip().capitalize()


def _find_entry(class_name: str, info: dict):
    """Exact normalized match first, then the longest knowledge-base key contained in the class name."""
    norm = _normalize(class_name)
    if norm in info:
        return info[norm]
    candidates = [k for k in info if not k.startswith("_") and (k in norm or norm in k)]
    if candidates:
        return info[max(candidates, key=len)]
    return None


def seed_diseases(db: Session, class_names: list[str]) -> list[str]:
    """Create/refresh one row per class. Returns class names that had no specific entry (fallback used)."""
    with open(DISEASE_INFO_PATH, encoding="utf-8") as f:
        info = json.load(f)
    fallback = info["_fallback"]

    missing = []
    for name in class_names:
        entry = _find_entry(name, info)
        if entry is None:
            entry = fallback
            missing.append(name)

        row = db.query(Disease).filter(Disease.name == name).first()
        if row is None:
            row = Disease(name=name)
            db.add(row)
        row.display_name = entry.get("display_name") or _pretty(name)
        row.severity = entry.get("severity", "unknown")
        row.description = entry.get("description", "")
        row.symptoms = entry.get("symptoms", "")
        row.treatment = entry.get("treatment", "")
        row.prevention = entry.get("prevention", "")
    db.commit()
    return missing
