import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def load_puzzles(path):
    df = pd.read_csv(path, encoding="latin-1")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Weekday"] = df["Date"].dt.day_name()
    df = df[(df["Date"].dt.year >= 2003) & (df["Weekday"] != "Sunday")]
    df = df.dropna(subset=["Clue", "Word"])
    df["Word"] = df["Word"].str.upper()
    return df[["Date", "Weekday", "Clue", "Word"]].reset_index(drop=True)
