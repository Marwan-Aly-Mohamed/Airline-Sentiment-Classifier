"""Inspect misclassified test tweets to find patterns worth fixing."""
import joblib
import pandas as pd

pd.set_option("display.max_colwidth", 90)

model = joblib.load("outputs/sentiment_model.joblib")
test = pd.read_csv("outputs/test_data.csv")

test["predicted"] = model.predict(test["clean_text"])
errors = test[test["label"] != test["predicted"]]

print(f"Total errors: {len(errors)} of {len(test)}\n")
print("Most common (true -> predicted) mistakes:")
print(errors.groupby(["label", "predicted"]).size().sort_values(ascending=False), "\n")

# Look at 10 random mistakes of the most important type
for true_label, pred_label in [("negative", "neutral"), ("neutral", "negative")]:
    subset = errors[(errors["label"] == true_label) & (errors["predicted"] == pred_label)]
    print(f"--- True: {true_label} | Predicted: {pred_label} ---")
    print(subset["clean_text"].sample(min(10, len(subset)), random_state=1).to_string(), "\n")