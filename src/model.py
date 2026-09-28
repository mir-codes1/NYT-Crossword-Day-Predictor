import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import WEEKDAYS
from src.features import NUMERIC_FEATURES, add_features

INPUT_COLUMNS = ["Clue", "Word"] + NUMERIC_FEATURES


def build_clue_model():
    preprocess = ColumnTransformer([
        ("clue", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "Clue"),
        ("answer", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True), "Word"),
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
    ])
    return Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=2000, C=0.3))])


def puzzle_profile(clue_probs, dates, classes):
    # average log-probability per weekday across a puzzle's clues
    logp = pd.DataFrame(np.log(np.clip(clue_probs, 1e-12, None)), columns=classes)
    logp["Date"] = np.asarray(dates)
    return logp.groupby("Date").mean()


class PuzzleDayPredictor:
    def fit(self, df):
        df = add_features(df)
        X, y, groups = df[INPUT_COLUMNS], df["Weekday"], df["Date"]

        self.clue_model = build_clue_model()
        # out-of-fold clue predictions so the puzzle model never sees scores from puzzles the clue model trained on
        oof = cross_val_predict(self.clue_model, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")
        self.clue_model.fit(X, y)
        self.classes_ = list(self.clue_model.classes_)

        profiles = puzzle_profile(oof, groups, self.classes_)
        puzzle_days = df.groupby("Date")["Weekday"].first().loc[profiles.index]
        self.puzzle_model = LogisticRegression(max_iter=2000).fit(profiles, puzzle_days)
        return self

    def _clue_probs(self, df):
        return self.clue_model.predict_proba(add_features(df)[INPUT_COLUMNS])

    def predict_puzzle(self, clues):
        # clues: DataFrame with Clue and Word columns, all from one puzzle
        profile = puzzle_profile(self._clue_probs(clues), np.zeros(len(clues)), self.classes_)
        probs = self.puzzle_model.predict_proba(profile)[0]
        return pd.Series(probs, index=self.puzzle_model.classes_).reindex(WEEKDAYS)

    def predict_clue(self, clue, answer):
        probs = self._clue_probs(pd.DataFrame({"Clue": [clue], "Word": [answer.upper()]}))[0]
        return pd.Series(probs, index=self.classes_).reindex(WEEKDAYS)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
