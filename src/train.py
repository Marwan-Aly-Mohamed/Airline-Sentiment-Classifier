"""Train the sentiment classifier: TF-IDF + a linear model."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

RANDOM_STATE = 42  # makes results reproducible


def build_pipeline(classifier) -> Pipeline:
    """TF-IDF vectorizer followed by a classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", classifier),
    ])


def main():
    # 1. Load data
    df = pd.read_csv("data/Tweets.csv")[["text", "airline_sentiment"]]

    # 2. Preprocess (stop-words + lemmatization)
    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]  # drop tweets that became empty

    X = df["clean_text"]
    y = df["airline_sentiment"]

    # 3. Stratified train/test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 4. Compare candidate models with 5-fold cross-validation (macro F1)
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_score = None, -1.0
    for name, clf in candidates.items():
        scores = cross_val_score(
            build_pipeline(clf), X_train, y_train, cv=cv, scoring="f1_macro"
        )
        print(f"{name}: macro F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    print(f"\nBest model: {best_name}")

    # 5. Train the best model on the full training set
    final_model = build_pipeline(candidates[best_name])
    final_model.fit(X_train, y_train)

    # 6. Save the model and the held-out test set for evaluate.py
    pd.DataFrame({"clean_text": X_train, "label": y_train}).to_csv("outputs/train_data.csv", index=False)
    joblib.dump(final_model, "outputs/sentiment_model.joblib")
    pd.DataFrame({"clean_text": X_test, "label": y_test}).to_csv(
        "outputs/test_data.csv", index=False
    )
    print("Saved: sentiment_model.joblib, train_data.csv, test_data.csv (in outputs/)")


if __name__ == "__main__":
    main()