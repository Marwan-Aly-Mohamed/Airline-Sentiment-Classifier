"""Final model: word TF-IDF + char TF-IDF + sentence embeddings + LogReg.

Chosen by 5-fold CV on the training set. Evaluated ONCE on the test set.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from evaluate import plot_confusion_matrix, plot_f1_per_class

LABELS = ["negative", "neutral", "positive"]


def make_frame(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Combine the cleaned text column with the embedding columns."""
    assert len(df) == len(embeddings), "Rows and embeddings do not match"
    frame = pd.DataFrame(
        embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])]
    )
    frame["clean_text"] = df["clean_text"].values
    return frame


train = pd.read_csv("outputs/train_data.csv")
test = pd.read_csv("outputs/test_data.csv")
X_train = make_frame(train, np.load("outputs/emb_train.npy"))
X_test = make_frame(test, np.load("outputs/emb_test.npy"))
y_train, y_test = train["label"], test["label"]

emb_cols = [c for c in X_train.columns if c != "clean_text"]

features = ColumnTransformer([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                             max_df=0.9, sublinear_tf=True), "clean_text"),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                             min_df=2, sublinear_tf=True), "clean_text"),
    ("emb", "passthrough", emb_cols),
])
model = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(C=1, max_iter=2000, class_weight="balanced")),
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
weighted_f1 = f1_score(y_test, pred, average="weighted")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}\n")
print(classification_report(y_test, pred, labels=LABELS, digits=3))

report = classification_report(y_test, pred, labels=LABELS, output_dict=True)
with open("outputs/metrics_final.json", "w") as f:
    json.dump({"accuracy": accuracy, "macro_f1": macro_f1,
               "weighted_f1": weighted_f1, "per_class": report}, f, indent=2)

plot_confusion_matrix(y_test, pred, LABELS, "outputs/confusion_matrix_final.png")
plot_f1_per_class(report, LABELS, "outputs/f1_per_class_final.png")
joblib.dump(model, "outputs/final_model.joblib")
print("Saved: metrics_final.json, confusion_matrix_final.png, "
      "f1_per_class_final.png, final_model.joblib")