"""Hyperparameter tuning with cross-validated grid search (macro F1).

Uses the training split only. The test set stays untouched. """
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

train = pd.read_csv("outputs/train_data.csv")
X_train, y_train = train["clean_text"], train["label"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.5, 1, 3, 10],
}

search = GridSearchCV(
    pipe, param_grid, scoring="f1_macro",
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print("Best CV macro F1:", round(search.best_score_, 4))
print("Best params:", search.best_params_)

# Show the top 5 combinations so you can see how much each setting matters
results = pd.DataFrame(search.cv_results_)
cols = ["param_tfidf__ngram_range", "param_tfidf__min_df", "param_clf__C", "mean_test_score"]
print(results.sort_values("mean_test_score", ascending=False)[cols].head(5))