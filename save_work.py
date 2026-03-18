from pathlib import Path
import joblib

MODEL_DIR = Path.cwd() / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
joblib.dump(clf, MODEL_DIR / "merchant_logreg.joblib")

print("Saved TF-IDF vectorizer and classifier to /models/")