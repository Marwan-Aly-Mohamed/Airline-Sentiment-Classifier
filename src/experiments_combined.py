"""Combine word TF-IDF, char TF-IDF and sentence embeddings (5-fold CV)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

train = pd.read_csv("outputs/train_data.csv")
E = np.load("outputs/emb_train.npy")
assert len(train) == len(E), "Embeddings and training rows do not match"

emb_cols = [f"e{i}" for i in range(E.shape[1])]
X = pd.DataFrame(E, columns=emb_cols)
X["clean_text"] = train["clean_text"].values
y = train["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for C in [0.5, 1]:
    features = ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                 max_df=0.9, sublinear_tf=True), "clean_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True), "clean_text"),
        ("emb", "passthrough", emb_cols),
    ])
    model = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"Word + char + embeddings, C={C}: "
          f"{scores.mean():.4f} (+/- {scores.std():.4f})")