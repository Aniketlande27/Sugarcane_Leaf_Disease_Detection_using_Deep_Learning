from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Float, Integer, Boolean, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _now():
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_file: Mapped[str] = mapped_column(String(255))            # file name inside uploads/
    predicted_label: Mapped[str] = mapped_column(String(120), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    all_probs: Mapped[dict] = mapped_column(JSON)
    is_uncertain: Mapped[bool] = mapped_column(Boolean, default=False)

    # user feedback ("was this correct?") - useful as new labelled data later
    feedback_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    feedback_label: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Disease(Base):
    __tablename__ = "diseases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)   # exactly the model's class name
    display_name: Mapped[str] = mapped_column(String(160))
    severity: Mapped[str] = mapped_column(String(20), default="unknown")      # none | low | moderate | high | unknown
    description: Mapped[str] = mapped_column(Text, default="")
    symptoms: Mapped[str] = mapped_column(Text, default="")
    treatment: Mapped[str] = mapped_column(Text, default="")
    prevention: Mapped[str] = mapped_column(Text, default="")
