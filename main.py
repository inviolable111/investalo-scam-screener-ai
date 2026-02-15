import os
import json
import re
import joblib
import threading
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Für Rate-Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# =====================================================
# SETUP & SICHERHEIT
# =====================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API Key, den das Frontend mitschicken muss
SECRET_APP_KEY = "investalo-public-frontend" 

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Investalo ScamScreener API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Im Live-Betrieb durch ["https://www.investalo.de"] ersetzen
    allow_methods=["POST"],
    allow_headers=["*"],
)

MODEL_PATH = "scam_ml_model.joblib"
VECTORIZER_PATH = "scam_vectorizer.joblib"
TRAINING_DATA_FILE = "training_data.jsonl"

FEEDBACK_COUNTER = 0
TRAIN_THRESHOLD = 10
model_lock = threading.Lock() 

# =====================================================
# ML MODEL LOGIK
# =====================================================
class ScamMLModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load(self):
        with model_lock: 
            if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                try:
                    self.model = joblib.load(MODEL_PATH)
                    self.vectorizer = joblib.load(VECTORIZER_PATH)
                    print("ML Modell & Vectorizer geladen.")
                except Exception as e:
                    print(f"Ladefehler: {e}")

    def predict(self, text: str) -> float:
        with model_lock: 
            if not self.model or not self.vectorizer:
                return 0.5
            try:
                X = self.vectorizer.transform([text])
                return float(self.model.predict_proba(X)[0][1])
            except:
                return 0.5

ml_model = ScamMLModel()
ml_model.load()

# =====================================================
# BACKGROUND TRAINING
# =====================================================
def run_background_training():
    global ml_model
    if not os.path.exists(TRAINING_DATA_FILE): return

    texts, labels = [], []
    with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                texts.append(data["text"])
                labels.append(1 if data["label"] == "scam" else 0)
            except: continue

    if len(set(labels)) < 2: 
        print("Training abgebrochen: Nicht genug verschiedene Labels (Scam & Safe benötigt).")
        return

    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    X = vec.fit_transform(texts)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, labels)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)
    ml_model.load()
    print("Background-Training abgeschlossen und Modell neu geladen.")

# =====================================================
# HELPERS
# =====================================================
def normalize_recommendations(raw):
    cleaned = []
    if not isinstance(raw, list) or len(raw) == 0:
        return [{"priority": "Info", "action": "Keine verdächtigen Links anklicken oder Anhänge öffnen."}]
    
    for r in raw:
        if not isinstance(r, dict): continue
        action = r.get("action", "").strip()
        priority = r.get("priority", "Mittel")
        if len(action) >= 2:
            cleaned.append({"priority": priority, "action": action})
            
    if not cleaned:
        cleaned.append({"priority": "Hinweis", "action": "Vorsicht bei unaufgeforderter Kontaktaufnahme."})
        
    return cleaned

# =====================================================
# API ENDPOINTS
# =====================================================
class AnalyzeRequest(BaseModel):
    content: str

@app.post("/analyze")
@limiter.limit("10/minute") 
async def analyze(request: Request, req: AnalyzeRequest, x_app_key: str = Header(None)):
    if x_app_key != SECRET_APP_KEY:
        raise HTTPException(status_code=403, detail="Ungültiger API-Key.")

    ml_risk = ml_model.predict(req.content)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Du bist der Investalo ScamScreener. Analysiere Nachrichten auf Betrug. "
                        "Antworte IMMER im JSON-Format mit: "
                        "score (0-100), risk (Niedrig|Mittel|Hoch), summary, "
                        "indicators (Liste), explanation, recommendations (Liste mit priority & action)."
                    )
                },
                {"role": "user", "content": req.content}
            ]
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"KI Fehler: {e}")
        data = {"score": 50, "summary": "Fehler bei der KI-Analyse."}

    ai_score = int(data.get("score", 50))
    # Kombination aus KI (70%) und lokalem ML-Modell (30%)
    final_score = min(100, int((0.7 * ai_score) + (0.3 * (ml_risk * 100))))

    return {
        "score": final_score,
        "risk": data.get("risk", "Mittel"),
        "summary": data.get("summary", ""),
        "indicators": data.get("indicators", []),
        "explanation_text": data.get("explanation", ""),
        "recommendations": normalize_recommendations(data.get("recommendations", [])),
        "ml_risk_score": ml_risk
    }

@app.post("/feedback-eval")
async def feedback_eval(req: dict, x_app_key: str = Header(None)):
    if x_app_key != SECRET_APP_KEY:
        raise HTTPException(status_code=403, detail="Ungültiger API-Key.")
    
    global FEEDBACK_COUNTER
    text = req.get("content", "")
    helpful = req.get("helpful", False) # Kommt als true/false vom Frontend

    if len(text) < 5:
        return {"status": "ignored"}

    # Logik zur Label-Erstellung für das Training:
    # Wir schauen, was das ML-Modell aktuell denkt:
    current_ml_score = ml_model.predict(text)
    
    # Wenn der User sagt "Hilfreich", bestätigen wir die aktuelle Tendenz.
    # Wenn der User sagt "Nicht hilfreich", kehren wir das Label um.
    if helpful:
        label = "scam" if current_ml_score >= 0.5 else "safe"
    else:
        label = "safe" if current_ml_score >= 0.5 else "scam"

    # In Datei schreiben
    with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    
    FEEDBACK_COUNTER += 1
    print(f"Feedback gespeichert. Counter: {FEEDBACK_COUNTER}/{TRAIN_THRESHOLD}")
    
    if FEEDBACK_COUNTER >= TRAIN_THRESHOLD:
        FEEDBACK_COUNTER = 0
        threading.Thread(target=run_background_training).start()

    return {"status": "ok", "assigned_label": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)