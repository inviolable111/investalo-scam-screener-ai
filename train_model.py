import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Pfade wie in deiner main.py
MODEL_PATH = "scam_ml_model.joblib"
VECTORIZER_PATH = "scam_vectorizer.joblib"
TRAINING_DATA_FILE = "training_data.jsonl"

def train():
    if not os.path.exists(TRAINING_DATA_FILE):
        print("Fehler: Keine Trainingsdaten gefunden (training_data.jsonl fehlt).")
        return

    texts = []
    labels = []

    # 1. Daten aus der JSONL laden
    with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            # Wir wandeln "scam" in 1 und "safe" in 0 um
            labels.append(1 if data["label"] == "scam" else 0)

    if len(set(labels)) < 2:
        print("Fehler: Du brauchst mindestens einen Scam-Fall und einen Safe-Fall zum Trainieren.")
        return

    print(f"Starte Training mit {len(texts)} Beispielen...")

    # 2. Vektorisierung (Text in Zahlen umwandeln)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)

    # 3. Das Modell trainieren (Classifier statt nur Anomalie-Erkennung)
    # Ein RandomForest lernt gezielte Muster aus deinen gelabelten Daten
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)

    # 4. Speichern
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Fertig! Das Modell wurde erfolgreich aktualisiert.")

if __name__ == "__main__":
    train()