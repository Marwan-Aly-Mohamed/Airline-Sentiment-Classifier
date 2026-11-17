"""Compare feature-engineering experiments with 5-fold CV (macro F1).

Uses the training split only. The test set stays untouched."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

train = pd.read_csv("outputs/train_data.csv")
X_train, y_train = train["clean_text"], train["label"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def make_model(features, C):
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


experiments = {
    "Word TF-IDF only (tuned)": make_model(word_tfidf(), C=3),
    "Word + char TF-IDF, C=1": make_model(
        FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())]), C=1),
    "Word + char TF-IDF, C=3": make_model(
        FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())]), C=3),
}

for name, model in experiments.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")