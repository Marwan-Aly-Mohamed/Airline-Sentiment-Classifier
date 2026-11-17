"""Quick exploratory look at the dataset."""
import pandas as pd

df = pd.read_csv("data/Tweets.csv")

# Keep only the two columns we need: the text and its label
df = df[["text", "airline_sentiment"]]  

print("Shape (rows, columns):", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution (%):\n", (df["airline_sentiment"].value_counts(normalize=True) * 100).round(1))
print("\nSample tweets:\n", df.sample(5, random_state=42).to_string())