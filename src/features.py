import numpy as np
from wordfreq import tokenize, zipf_frequency


def clue_word_scores(clue):
    # wordfreq keeps "___" as a token; drop anything without a letter
    words = [w for w in tokenize(clue, "en") if any(ch.isalpha() for ch in w)]
    return [zipf_frequency(w, "en") for w in words]


def avg_word_frequency(clue):
    scores = clue_word_scores(clue)
    return float(np.mean(scores)) if scores else np.nan


def min_word_frequency(clue):
    scores = clue_word_scores(clue)
    return min(scores) if scores else np.nan


def answer_frequency(answer):
    return zipf_frequency(answer.lower(), "en")


NUMERIC_FEATURES = [
    "Clue Length", "Avg Word Length", "Has Fill Blank", "Has Question Mark",
    "Avg Word Freq", "Min Word Freq", "Answer Freq", "Answer Length",
]


def add_features(df):
    df = df.copy()
    df["Clue Length"] = df["Clue"].str.len()
    df["Avg Word Length"] = df["Clue Length"] / (df["Clue"].str.count(" ") + 1)
    df["Has Fill Blank"] = df["Clue"].str.contains("_", regex=False)
    df["Has Question Mark"] = df["Clue"].str.contains("?", regex=False)
    df["Avg Word Freq"] = df["Clue"].apply(avg_word_frequency)
    df["Min Word Freq"] = df["Clue"].apply(min_word_frequency)
    df["Answer Freq"] = df["Word"].apply(answer_frequency)
    df["Answer Length"] = df["Word"].str.len()
    return df
