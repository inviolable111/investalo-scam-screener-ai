import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Pfade wie in der main.py
MODEL_PATH = "scam_ml_model.joblib"
VECTORIZER_PATH = "scam_vectorizer.joblib"
TRAINING_DATA_FILE = "training_data.jsonl"

def train():
    if not os.path.exists(TRAINING_DATA_FILE):
        print(f"Fehler: Keine Trainingsdaten gefunden ({TRAINING_DATA_FILE} fehlt).")
        return

    texts = []
    labels = []

    # 1. Daten aus der JSONL laden
    print("Lade Daten aus training_data.jsonl...")
    with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Wir unterstützen 'text' (Standard) oder 'content' (vom Frontend)
                txt = data.get("text") or data.get("content")
                lbl = data.get("label")
                
                if txt and lbl:
                    texts.append(txt)
                    labels.append(1 if lbl == "scam" else 0)
            except Exception as e:
                continue

    if len(set(labels)) < 2:
        print("Fehler: Du brauchst mindestens einen Scam-Fall und einen Safe-Fall zum Trainieren.")
        print(f"Aktuell geladen: {len(labels)} Zeilen.")
        return

    print(f"Starte Training mit {len(texts)} Beispielen...")

    # 2. Vektorisierung (Text in Zahlen umwandeln)
    # ngram_range(1,3) lernt auch Wortkombinationen
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)

    # 3. Das Modell trainieren
    # RandomForest ist sehr robust für Textklassifizierung
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)

    # 4. Speichern
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("-" * 30)
    print("ERFOLG!")
    print(f"Modell gespeichert unter: {MODEL_PATH}")
    print(f"Vectorizer gespeichert unter: {VECTORIZER_PATH}")
    print("Das System ist nun auf dem neuesten Stand.")
    print("-" * 30)

if __name__ == "__main__":
    train()