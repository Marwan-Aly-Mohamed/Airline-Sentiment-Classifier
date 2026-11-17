"""Predict the sentiment of new text with the trained final model.

Usage (from the project root):
    python src/predict.py "My flight was cancelled again"
    python src/predict.py            # interactive mode
"""
import sys

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocess import clean_text, light_clean

MODEL_PATH = "outputs/final_model.joblib"
ENCODER_NAME = "all-MiniLM-L6-v2"


def predict(texts, model, encoder):
    """Return (label, confidence) for each input text."""
    embeddings = encoder.encode([light_clean(t) for t in texts])
    frame = pd.DataFrame(
        embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])]
    )
    frame["clean_text"] = [clean_text(t) for t in texts]

    probabilities = model.predict_proba(frame)
    best = probabilities.argmax(axis=1)
    return [(model.classes_[i], probabilities[row, i])
            for row, i in enumerate(best)]


def main():
    model = joblib.load(MODEL_PATH)
    encoder = SentenceTransformer(ENCODER_NAME)

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        print("Type a sentence (empty line to quit).")
        texts = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            label, conf = predict([line], model, encoder)[0]
            print(f"  -> {label} ({conf:.0%})")
        return

    for text, (label, conf) in zip(texts, predict(texts, model, encoder)):
        print(f"{label} ({conf:.0%}): {text}")


if __name__ == "__main__":
    main()