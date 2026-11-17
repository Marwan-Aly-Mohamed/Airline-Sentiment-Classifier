"""Sentence-embedding + Logistic Regression classifier (comparison model).

Uses the SAME train/test split as train.py so the scores are comparable."""
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from preprocess import clean_text, light_clean

RANDOM_STATE = 42


def light_clean(text: str) -> str:
    """Remove only URLs and @mentions; keep everything else for the model."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# 1. Rebuild the exact same split as train.py
df = pd.read_csv("data/Tweets.csv")[["text", "airline_sentiment"]]
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 0]          # same filter as train.py
df["embed_text"] = df["text"].apply(light_clean)

X_train, X_test, y_train, y_test = train_test_split(
    df["embed_text"], df["airline_sentiment"],
    test_size=0.2, stratify=df["airline_sentiment"], random_state=RANDOM_STATE,
)

# Safety check: same test tweets, same order as the TF-IDF experiments
saved_test = pd.read_csv("outputs/test_data.csv")
assert (df.loc[X_test.index, "clean_text"].values
        == saved_test["clean_text"].values).all(), "Split does not match train.py!"

# 2. Turn tweets into vectors (cached to disk after the first run)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts, path):
    if os.path.exists(path):
        return np.load(path)
    vectors = model.encode(texts.tolist(), batch_size=64, show_progress_bar=True)
    np.save(path, vectors)
    return vectors


E_train = embed(X_train, "outputs/emb_train.npy")
E_test = embed(X_test, "outputs/emb_test.npy")
print("Embedding shape (rows, dimensions):", E_train.shape)

# 3. Choose C with cross-validation on the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_C, best_score = None, -1.0
for C in [0.5, 1, 3, 10]:
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    score = cross_val_score(clf, E_train, y_train, cv=cv, scoring="f1_macro").mean()
    print(f"C={C}: CV macro F1 = {score:.4f}")
    if score > best_score:
        best_C, best_score = C, score

# 4. Final fit and a single evaluation on the held-out test set
final = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
final.fit(E_train, y_train)
pred = final.predict(E_test)

print(f"\nBest C: {best_C}")
print(f"Test macro F1: {f1_score(y_test, pred, average='macro'):.4f}\n")
print(classification_report(y_test, pred, digits=3))