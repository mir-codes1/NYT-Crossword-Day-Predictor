# NYT Crossword Day Predictor

The New York Times crossword gets harder as the week goes on: Monday is the easiest, Saturday the hardest. This project trains a model that reads a puzzle's clues and answers and guesses which day of the week it ran, in effect estimating difficulty from the text alone.

A Streamlit app lets you try it:

- **Guess a puzzle's day:** pull a random held-out puzzle, read its clues, then reveal the model's guess, the actual date, and the probability for each day.
- **Try your own clue:** enter a single clue and answer to see how the model rates it.

## Results

Evaluated on 1,180 held-out puzzles (20% of puzzles, split by date so no puzzle appears in both train and test):

| Metric | Accuracy |
|---|---|
| Exact day, full puzzle | **62%** (random guessing: 17%) |
| Within one day, full puzzle | **97%** |
| Exact day, single clue | 24% |

Almost all mistakes land on an adjacent day. Monday is the easiest to identify; Friday and Saturday are the pair most often confused.

```
pred       Monday  Tuesday  Wednesday  Thursday  Friday  Saturday
true
Monday        179       13          0         0       0         0
Tuesday        44      111         55        10       0         0
Wednesday       3       40         99        32       1         0
Thursday        0        2         39       140      21         7
Friday          0        0          0        31      70        89
Saturday        0        0          1         9      49       135
```

## Getting started

Requires Python 3.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# launch the demo (uses the trained model in models/)
streamlit run app.py

# optional: retrain the model and print the evaluation
python -m src.train
```

## Data

`data/nytcrosswords.csv` contains about 781k clues (`Date, Word, Clue`) from NYT puzzles published between November 1993 and October 2021.

The pipeline keeps only:

- **Puzzles from 2003 onward.** Before about 2000 the Monday-to-Saturday difficulty ramp is weak; the rate of `?` clues (usually puns) only climbs steadily through the week in later years.
- **Monday through Saturday.** Sunday puzzles are larger and sit around Wednesday/Thursday difficulty, so they break the pattern.

That leaves about 401k clues across 5,896 puzzles, around 68 clues each.

## How it works

A single clue carries very little signal, but a puzzle has about 70 of them. The model scores each clue, then pools the evidence across the whole puzzle.

**Stage 1: clue model.** A logistic regression predicts the day from one clue/answer pair, using:

- word 1–2-grams of the clue (TF-IDF)
- character 2–4-grams of the answer (TF-IDF), which pick up obscure or unusual fill
- numeric features: clue length, average word length, whether the clue has a fill-in-the-blank (`___`), whether it has a question mark, the average and minimum English word frequency of the clue's words (via [`wordfreq`](https://github.com/rspeer/wordfreq)), the answer's frequency, and the answer's length

**Stage 2: puzzle model.** For each puzzle, the clue model's log-probabilities are averaged per weekday into a 6-number profile, and a second logistic regression maps that profile to a day.

To avoid leakage, stage 2 is trained on out-of-fold stage 1 predictions (`GroupKFold` grouped by puzzle date), so it never sees scores for puzzles the clue model was trained on. The app only shows puzzles from the held-out test set.

## Exploratory analysis

`notebooks/01_eda.ipynb` covers the reasoning behind these choices: checking that the difficulty ramp holds over time, measuring each candidate feature's association with the weekday (chi-square / Cramér's V, ANOVA / η²), and finding that surface-structure features alone barely beat a baseline (19% vs 17% per clue). That result motivated adding word-frequency features and text n-grams, and scoring whole puzzles instead of single clues.

## Project structure

```
app.py                              Streamlit demo
src/data.py                         Loading and filtering the puzzle data
src/features.py                     Hand-built clue and answer features
src/model.py                        Two-stage PuzzleDayPredictor
src/train.py                        Train/test split, training, evaluation, saving
models/puzzle_day_predictor.joblib  Trained model
models/demo_puzzle_dates.csv        Held-out puzzle dates shown in the app
notebooks/01_eda.ipynb              Exploratory analysis
data/nytcrosswords.csv              Clue dataset
```

## Limitations

- Crosswords reuse a lot of fill (e.g. "Stadium sound" → RAH). The train/test split is by puzzle, but identical clue/answer pairs can still appear in both sets, which may flatter the reported accuracy somewhat.
- Sunday puzzles and puzzles from before 2003 aren't supported.
- Single-clue predictions are weak (24%) and should be read as a rough signal only.
