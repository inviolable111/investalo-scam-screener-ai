<p align="center">
  <img src="preview.png" alt="Screenshot" width="80%">
</p>
# <img src="Maskottchen.png" width="150" align="right"> Investalo ScamScreener: Don't get rekt! 🛡️

Hand aufs Herz: Der Finanzmarkt ist voll von Scams. Mit **Investalo** baue ich Tools, die dem Nutzer den Rücken freihalten. Der **ScamScreener** ist mein erster Bodyguard gegen manipulative Nachrichten und "Schnell-reich-werden"-Müll.

## 🧠 Warum dieser Ansatz?
Ich wollte nicht nur eine einfache KI-Abfrage. Deshalb habe ich ein **hybrides System** gebaut:
1. **OpenAI (The Brain):** Versteht den Kontext, die Psychologie hinter dem Scam und die fiesen Tricks der Betrüger.
2. **Local ML (The Guard):** Ein Random Forest Modell, das direkt auf meinem Code läuft. Es lernt aus jedem Fall dazu und erkennt statistische Muster, die man mit bloßem Auge übersieht.

## 🔥 Tech-Highlights
* **FastAPI Backend:** Schlank, schnell und asynchron – für Performance ohne Wartezeit.
* **Hybrid Scoring:** 70% KI-Expertise + 30% lokales Machine Learning für maximale Treffsicherheit.
* **Self-Improving:** Dank des `train_model.py` Skripts lernt der Screener mit jeder neuen Scam-Welle dazu.
* **Modern UI:** Ein sauberes Interface im Glassmorphism-Style, das Vertrauen schafft und einfach zu bedienen ist.

## 🔧 Installation & Start
1. Repository klonen.
2. `pip install -r requirements.txt`
3. Erstelle eine `.env` Datei und füge deinen `OPENAI_API_KEY` ein.
4. Starte die App mit: `uvicorn main:app --reload`

---
**Investalo – Empowering your financial journey with code.**
