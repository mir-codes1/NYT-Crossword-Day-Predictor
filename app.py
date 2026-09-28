from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.data import WEEKDAYS, load_puzzles
from src.model import PuzzleDayPredictor

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="NYT Crossword Day Predictor", layout="centered")


@st.cache_resource
def get_model():
    return PuzzleDayPredictor.load(ROOT / "models" / "puzzle_day_predictor.joblib")


@st.cache_data
def get_demo_puzzles():
    dates = pd.to_datetime(pd.read_csv(ROOT / "models" / "demo_puzzle_dates.csv")["Date"])
    df = load_puzzles(ROOT / "data" / "nytcrosswords.csv")
    return df[df["Date"].isin(dates)]


def probability_chart(probs):
    dark = st.context.theme.type == "dark"
    bar_color, text_color = ("#3987e5", "#c3c2b7") if dark else ("#2a78d6", "#52514e")
    data = pd.DataFrame({"Day": probs.index, "Probability": probs.values})
    base = alt.Chart(data).encode(
        y=alt.Y("Day:N", sort=WEEKDAYS, title=None),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5), title=None),
        tooltip=[alt.Tooltip("Day:N"), alt.Tooltip("Probability:Q", format=".1%")],
    )
    bars = base.mark_bar(color=bar_color, size=20, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=6, color=text_color).encode(text=alt.Text("Probability:Q", format=".0%"))
    return (bars + labels).properties(height=200)


st.title("NYT Crossword Day Predictor")
st.write("NYT crosswords get harder as the week goes on (Monday is easiest, Saturday hardest). Given a puzzle's clues and answers, the model guesses what day it ran.")

puzzle_tab, clue_tab = st.tabs(["Guess a puzzle's day", "Try your own clue"])

with puzzle_tab:
    puzzles = get_demo_puzzles()
    st.caption(f"Pulled from {puzzles['Date'].nunique():,} held-out puzzles. Test accuracy: 62% exact day, 97% within a day.")

    if st.button("Pick a random puzzle", type="primary"):
        st.session_state.puzzle_date = puzzles["Date"].drop_duplicates().sample(1).iloc[0]
        st.session_state.revealed = False

    if "puzzle_date" in st.session_state:
        puzzle = puzzles[puzzles["Date"] == st.session_state.puzzle_date]
        st.subheader(f"{len(puzzle)} clues")
        st.dataframe(puzzle[["Clue", "Word"]].rename(columns={"Word": "Answer"}), hide_index=True, height=260)

        if st.button("Reveal the model's guess"):
            st.session_state.revealed = True

        if st.session_state.get("revealed"):
            probs = get_model().predict_puzzle(puzzle)
            predicted, actual = probs.idxmax(), puzzle["Weekday"].iloc[0]
            days_off = abs(WEEKDAYS.index(predicted) - WEEKDAYS.index(actual))

            left, right = st.columns(2)
            left.metric("Model's guess", predicted, f"{probs.max():.0%} confident", delta_color="off", delta_arrow="off")
            right.metric("Actual day", actual, st.session_state.puzzle_date.strftime("%B %-d, %Y"), delta_color="off", delta_arrow="off")

            st.write("Got it right." if days_off == 0 else f"Off by {days_off} day{'s' if days_off > 1 else ''}.")

            st.altair_chart(probability_chart(probs), width="stretch")

with clue_tab:
    st.caption("One clue doesn't tell you much. Single-clue accuracy is only ~24%, vs 62% for a full puzzle.")
    with st.form("clue_form"):
        clue = st.text_input("Clue", placeholder="Low-lying areas?")
        answer = st.text_input("Answer", placeholder="HIDEOUTS")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        if not clue.strip() or not answer.strip():
            st.error("Enter both a clue and an answer.")
        else:
            probs = get_model().predict_clue(clue.strip(), answer.strip().replace(" ", ""))
            st.metric("Most likely day", probs.idxmax(), f"{probs.max():.0%} probability", delta_color="off", delta_arrow="off")
            st.altair_chart(probability_chart(probs), width="stretch")
