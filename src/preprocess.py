"""Text preprocessing: cleaning, stop-word removal, and lemmatization."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data 
for package in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(package, quiet=True)

# Words that flip sentiment must NOT be removed
NEGATIONS = {"no", "nor", "not", "never"}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Convert a raw tweet into a clean, lemmatized string."""
    text = text.lower()

    # Expand contractions so negation survives punctuation removal
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)

    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"@\w+", " ", text)              # remove @mentions
    text = re.sub(r"#", " ", text)                 # drop '#', keep the word
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def light_clean(text: str) -> str:
    """Remove only URLs and @mentions (used for the embedding model)."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if __name__ == "__main__":
    examples = [
        "@united my flights were delayed AGAIN!!! Not happy :( http://t.co/xyz",
        "Thanks @SouthwestAir, the crew was amazing! #bestflight",
    ]
    for e in examples:
        print("RAW  :", e)
        print("CLEAN:", clean_text(e), "\n")
