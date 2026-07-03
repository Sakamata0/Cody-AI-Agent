"""
Train a Prophet model to predict monthly tornado frequency by US state.

Uses NOAA SPC historical tornado data (1950-2023).
Saves trained models as pickle files for use by the prediction tool.
"""

import os
import pickle
import pandas as pd
from prophet import Prophet

DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DATA_DIR, "models")


def train_state_model(df: pd.DataFrame, state: str) -> Prophet:
    """Train a Prophet model for a specific state."""
    # Filter by state and aggregate monthly counts.
    state_df = df[df["st"] == state].copy()
    state_df["date"] = pd.to_datetime(state_df[["yr", "mo", "dy"]].rename(
        columns={"yr": "year", "mo": "month", "dy": "day"}
    ))

    # Monthly tornado count.
    monthly = state_df.resample("MS", on="date").size().reset_index()
    monthly.columns = ["ds", "y"]

    # Train Prophet model.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(monthly)
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading tornado data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "tornadoes_raw.csv"), low_memory=False)
    print(f"Total records: {len(df)}")

    # Train models for the top tornado-prone states.
    # These are the states with the most tornado activity historically.
    top_states = ["TX", "KS", "OK", "FL", "NE", "IL", "MS", "AL", "IA", "MO"]

    for state in top_states:
        count = len(df[df["st"] == state])
        print(f"Training model for {state} ({count} records)...")
        try:
            model = train_state_model(df, state)
            model_path = os.path.join(MODELS_DIR, f"tornado_{state}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"  Saved: {model_path}")
        except Exception as e:
            print(f"  Error for {state}: {e}")

    # Also train a national model (all states combined).
    print("Training national model (all US)...")
    df["date"] = pd.to_datetime(df[["yr", "mo", "dy"]].rename(
        columns={"yr": "year", "mo": "month", "dy": "day"}
    ), errors="coerce")
    monthly_national = df.dropna(subset=["date"]).resample("MS", on="date").size().reset_index()
    monthly_national.columns = ["ds", "y"]

    national_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    national_model.fit(monthly_national)

    with open(os.path.join(MODELS_DIR, "tornado_US.pkl"), "wb") as f:
        pickle.dump(national_model, f)
    print("  Saved: tornado_US.pkl")

    print("\nTraining complete!")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
