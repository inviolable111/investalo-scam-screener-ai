import os
import json
import re
import joblib
import numpy as np
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# SETUP
# =====================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Investalo ScamScreener API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "scam_ml_model.joblib"
VECTORIZER_PATH = "scam_vectorizer.joblib"
TRAINING_DATA_FILE = "training_data.jsonl"

SCAM_KEYWORDS = [
    "paket", "zoll", "zustellung", "dhl", "fedex", "ups",
    "konto", "verifizieren", "tan", "gesperrt",
    "gewinn", "erbschaft", "krypto", "bitcoin",
    "rechnung", "mahnung", "inkasso", "polizei"
]

# =====================================================
# ML MODEL
# =====================================================

class ScamMLModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict(self, text: str) -> float:
        if not self.model or not self.vectorizer:
            return 0.5
        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0][1])

ml_model = ScamMLModel()
ml_model.load()

# =====================================================
# HELPER: Recommendation Handling
# =====================================================

def normalize_recommendations(raw):
    cleaned = []

    if not isinstance(raw, list):
        return cleaned

    for r in raw:
        if not isinstance(r, dict):
            continue

        action = r.get("action")
        priority = r.get("priority", "Mittel")

        if not action or len(action.strip()) < 10:
            continue

        if priority not in ["Hoch", "Mittel", "Niedrig"]:
            priority = "Mittel"

        cleaned.append({
            "priority": priority,
            "action": action.strip()
        })

    return cleaned


def fallback_recommendations(score: int) -> List[Dict]:
    if score >= 80:
        return [
            {"priority": "Hoch", "action": "Reagiere nicht auf die Nachricht und klicke auf keinen enthaltenen Link."},
            {"priority": "Hoch", "action": "Blockiere den Absender und lösche die Nachricht sofort."},
            {"priority": "Mittel", "action": "Falls du bereits reagiert hast, kontaktiere vorsorglich deine Bank oder den betroffenen Anbieter."}
        ]
    elif score >= 40:
        return [
            {"priority": "Mittel", "action": "Überprüfe den Absender genau und öffne keine Links oder Anhänge."},
            {"priority": "Mittel", "action": "Vergleiche die Nachricht mit offiziellen Informationen des angeblichen Absenders."}
        ]
    else:
        return [
            {"priority": "Niedrig", "action": "Die Nachricht wirkt überwiegend unauffällig – bleib dennoch aufmerksam."}
        ]

# =====================================================
# API
# =====================================================

class AnalyzeRequest(BaseModel):
    content: str


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    ml_risk = ml_model.predict(req.content)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Du bist der Investalo ScamScreener.
Antworte IMMER im JSON-Format mit folgenden Feldern:

score (0–100),
risk ("Niedrig" | "Mittel" | "Hoch"),
summary (1 kurzer Satz),
indicators (Liste konkreter Betrugsmerkmale),
explanation (kurze Erklärung für Laien),

recommendations:
- Array aus 3–5 Objekten
- Jedes Objekt:
  - priority: "Hoch" | "Mittel" | "Niedrig"
  - action: 1 konkreter, individueller Handlungssatz
- Beziehe dich direkt auf den Inhalt der Nachricht
- Keine allgemeinen Floskeln

user_feedback (1 beruhigender Satz)
"""
                },
                {"role": "user", "content": req.content}
            ]
        )

        data = json.loads(response.choices[0].message.content)

    except Exception:
        data = {}

    ai_score = int(data.get("score", 50))
    final_score = int((0.7 * ai_score) + (0.3 * (ml_risk * 100)))

    recommendations = normalize_recommendations(
        data.get("recommendations", [])
    )

    if len(recommendations) < 2:
        recommendations = fallback_recommendations(final_score)

    return {
        "score": min(100, final_score),
        "risk": data.get("risk", "Mittel"),
        "summary": data.get("summary", "Analyse abgeschlossen."),
        "indicators": data.get("indicators", []),
        "explanation_text": data.get("explanation", ""),
        "recommendations": recommendations,
        "user_feedback": data.get("user_feedback", "Gut, dass du vorsichtig bist."),
        "ml_risk_score": ml_risk
    }


# =====================================================
# START
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
