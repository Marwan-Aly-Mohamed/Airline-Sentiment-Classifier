"""Evaluate the trained model on the held-out test set using F1-scores."""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def plot_confusion_matrix(y_true, y_pred, labels, path):
    """Save a row-normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (normalized by true class)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_f1_per_class(report: dict, labels, path):
    """Save a bar chart of F1-score per class."""
    scores = [report[label]["f1-score"] for label in labels]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, scores, color=["#d9534f", "#f0ad4e", "#5cb85c"])
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("F1-score per class")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.02,
                 f"{score:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    model = joblib.load("outputs/sentiment_model.joblib")
    test = pd.read_csv("outputs/test_data.csv")

    X_test, y_test = test["clean_text"], test["label"]
    y_pred = model.predict(X_test)
    labels = ["negative", "neutral", "positive"]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}\n")
    print(classification_report(y_test, y_pred, labels=labels, digits=3))

    # Save results
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True
    )
    with open("outputs/metrics.json", "w") as f:
        json.dump(
            {"accuracy": accuracy, "macro_f1": macro_f1,
             "weighted_f1": weighted_f1, "per_class": report},
            f, indent=2,
        )

    plot_confusion_matrix(y_test, y_pred, labels, "outputs/confusion_matrix.png")
    plot_f1_per_class(report, labels, "outputs/f1_per_class.png")
    print("Saved: metrics.json, confusion_matrix.png, f1_per_class.png")


if __name__ == "__main__":
    main()