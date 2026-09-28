from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data import WEEKDAYS, load_puzzles
from src.model import PuzzleDayPredictor

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, test):
    rows = []
    for date, puzzle in test.groupby("Date"):
        probs = model.predict_puzzle(puzzle)
        rows.append((puzzle["Weekday"].iloc[0], probs.idxmax()))
    results = pd.DataFrame(rows, columns=["true", "pred"])
    distance = (results["true"].map(WEEKDAYS.index) - results["pred"].map(WEEKDAYS.index)).abs()
    print(f"Held-out puzzles: {len(results)}")
    print(f"Exact day accuracy:   {(distance == 0).mean():.3f}")
    print(f"Within one day:       {(distance <= 1).mean():.3f}")
    print(pd.crosstab(results["true"], results["pred"]).reindex(index=WEEKDAYS, columns=WEEKDAYS, fill_value=0))


def main():
    df = load_puzzles(ROOT / "data" / "nytcrosswords.csv")
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(df, groups=df["Date"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    model = PuzzleDayPredictor().fit(train)
    evaluate(model, test)

    # app only shows test puzzles
    model.save(ROOT / "models" / "puzzle_day_predictor.joblib")
    pd.Series(sorted(test["Date"].unique())).to_csv(ROOT / "models" / "demo_puzzle_dates.csv", index=False, header=["Date"])
    print("Saved models/puzzle_day_predictor.joblib and models/demo_puzzle_dates.csv")


if __name__ == "__main__":
    main()
