from pathlib import Path
import re
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "spam_model.joblib"

app = FastAPI(
    title="Spam Classification API",
    description="API для бинарной классификации писем: spam или ham",
    version="1.0.0",
)

try:
    artifact = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

vectorizer = artifact["vectorizer"]
model = artifact["model"]
threshold = artifact["threshold"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст письма: тема и тело")


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    spam_probability: float
    needs_human_review: bool


def prepare_features(text: str):
    subject = text.split("\n", 1)[0]
    body = text.split("\n", 1)[1] if "\n" in text else ""

    manual = np.array([[
        len(subject),
        sum(1 for c in subject if c.isupper()),
        subject.count("!"),
        len(body),
        len(re.findall(r"http[s]?://", text)),
        int(bool(re.search(r"<html|<div|<table", text, re.I))),
        text.count("!") / max(len(text), 1) * 1000,
    ]], dtype=float)

    return manual


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "spam-classification-api",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        text = request.text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Поле text не должно быть пустым",
            )

        X_text = vectorizer.transform([text])
        X_manual = prepare_features(text)

        from scipy.sparse import hstack, csr_matrix

        X = hstack([X_text, csr_matrix(X_manual)]).tocsr()

        spam_probability = float(model.predict_proba(X)[0, 1])
        label_id = int(spam_probability >= threshold)

        needs_human_review = 0.40 <= spam_probability <= 0.60

        return PredictResponse(
            label="spam" if label_id == 1 else "ham",
            label_id=label_id,
            confidence=spam_probability if label_id == 1 else 1 - spam_probability,
            spam_probability=spam_probability,
            needs_human_review=needs_human_review,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}",
        )