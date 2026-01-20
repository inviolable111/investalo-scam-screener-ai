import os
import json
import re
import joblib
import numpy as np
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 1. SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Investalo ScamScreener API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PFADE & KONFIGURATION
MODEL_PATH = "scam_ml_model.joblib"
VECTORIZER_PATH = "scam_vectorizer.joblib"
TRAINING_DATA_FILE = "training_data.jsonl"

SCAM_KEYWORDS = [
    "paket", "zoll", "zustellung", "dhl", "fedex", "ups", "abholen", "nachzahlung",
    "konto", "verifizieren", "tan", "push", "gesperrt", "einschränkung", "s-push", "sparkasse",
    "kind", "mama", "papa", "handy", "nummer", "geld", "notfall", "überweisen",
    "gewinn", "erbschaft", "krypto", "investition", "rendite", "bitco",
    "rechnung", "mahnung", "inkasso", "polizei", "staatsanwaltschaft", "haftbefehl"
]

# --- NEU: AUTOMATISCHES TRAINING ---
def auto_train_model():
    """Liest die JSONL und trainiert das Modell neu, falls Daten vorhanden sind."""
    if not os.path.exists(TRAINING_DATA_FILE):
        return
    
    texts, labels = [], []
    try:
        with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                texts.append(data["text"])
                labels.append(1 if data["label"] == "scam" else 0)
        
        if len(set(labels)) >= 2:
            print(f"🔄 Auto-Training: Aktualisiere Modell mit {len(texts)} Beispielen...")
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
            X = vectorizer.fit_transform(texts)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, labels)
            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            print("✅ Modell erfolgreich auto-aktualisiert.")
    except Exception as e:
        print(f"⚠️ Auto-Training Fehler: {e}")

# --- ML KLASSE ---
class ScamMLModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def _extra_features(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        return np.array([
            len(text),
            len(re.findall(r'https?://', text_lower)),
            len(re.findall(r'[0-9]{5,}', text_lower)),
            sum(c.isupper() for c in text) / max(len(text), 1),
            sum(1 for k in SCAM_KEYWORDS if k in text_lower),
            1 if any(x in text_lower for x in ["ae", "oe", "ue"]) and not any(x in text_lower for x in ["ä", "ö", "ü"]) else 0
        ])

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            return True
        return False

    def predict(self, text: str) -> float:
        if not self.model or not self.vectorizer:
            return 0.5
        try:
            # Check if it's the new RandomForest or old IsolationForest
            X_text = self.vectorizer.transform([text])
            if isinstance(self.model, RandomForestClassifier):
                # Gibt die Wahrscheinlichkeit für Klasse 1 (Scam) zurück
                return float(self.model.predict_proba(X_text)[0][1])
            else:
                # Fallback für alten IsolationForest
                score = self.model.decision_function(X_text)[0]
                return float(round(np.clip((1 - score) / 2, 0, 1), 3))
        except:
            return 0.5

# Initialisierung
auto_train_model() # Trainiere beim Starten kurz nach
ml_model = ScamMLModel()
ml_model.load()

# --- API ENDPUNKTE ---
class AnalyzeRequest(BaseModel):
    content: str

def log_case_for_learning(text: str, ai_data: Dict):
    score = int(ai_data.get("score", 50))
    if score > 85 or score < 15:
        label = "scam" if score > 85 else "safe"
        entry = {"text": text.replace("\n", " "), "label": label, "score": score}
        with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    ml_risk = ml_model.predict(req.content)
    
    # OpenAI Analyse
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Du bist der Investalo ScamScreener Bot. Antworte IMMER im JSON-Format (score, risk, summary, indicators, explanation, recommendations, user_feedback)."},
                {"role": "user", "content": f"Analysiere auf Betrug: {req.content}"}
            ]
        )
        data = json.loads(response.choices[0].message.content)
    except:
        data = {}

    log_case_for_learning(req.content, data)
    
    ai_score = int(data.get("score", 50))
    final_score = int((0.7 * ai_score) + (0.3 * (ml_risk * 100)))
    
    return {
        "score": min(100, final_score),
        "risk": data.get("risk", "Mittel"),
        "summary": data.get("summary", "Analyse abgeschlossen."),
        "indicators": data.get("indicators", []),
        "explanation_text": data.get("explanation", "Keine weiteren Details."),
        "recommendations": data.get("recommendations", []),
        "user_feedback": data.get("user_feedback", "Bleib wachsam!"),
        "ml_risk_score": ml_risk
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)