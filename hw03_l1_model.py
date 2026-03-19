
import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def print_top_words(model, vectorizer, top_n=15):
    coef = model.coef_[0]
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]

    print("\n Top Positive Words (CORE = 1) ")
    for i in top_pos_idx:
        print(f"{feature_names[i]} ({coef[i]:.4f})")

    print("\n Top Negative Words (NEG = 0) ")
    for i in top_neg_idx:
        print(f"{feature_names[i]} ({coef[i]:.4f})")

# -----------------------------
# 1. Load JSON Data
# -----------------------------
DATA_DIR = Path("data")

TRAIN_PATH = DATA_DIR / "train_core_vs_neg.json"
TEST_PATH = DATA_DIR / "test_core_vs_neg.json"

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_df = load_json(TRAIN_PATH)
test_df = load_json(TEST_PATH)

train_df.columns = ["text", "label"]
test_df.columns = ["text", "label"]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# -----------------------------
# 2. Define X and y
# -----------------------------
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# -----------------------------
# 3. TF-IDF 
# -----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF shape:", X_train_tfidf.shape)


# -----------------------------
# 4. Train Logistic Regression (L1)
# -----------------------------
model_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.1,        
    max_iter=2000
)

model_l1.fit(X_train_tfidf, y_train)


# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model_l1.predict(X_test_tfidf)
y_prob = model_l1.predict_proba(X_test_tfidf)[:, 1]


# -----------------------------
# 6. Evaluation
# -----------------------------
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC AUC ===")
print(roc_auc_score(y_test, y_prob))


# -----------------------------
# 7. Sparsity 
# -----------------------------
coef = model_l1.coef_[0]

num_nonzero = np.sum(coef != 0)
num_total = len(coef)

print("\n=== Model Sparsity (L1) ===")
print("Non-zero coefficients:", num_nonzero)
print("Total coefficients:", num_total)
print("Percent non-zero:", (num_nonzero / num_total) * 100)

print_top_words(model_l1, vectorizer)