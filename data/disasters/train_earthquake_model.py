"""
Train Prophet models to predict monthly earthquake frequency for Japan.

Uses USGS historical earthquake data (1950-2024), magnitude 4.5+.
"""

import os
import pickle
import pandas as pd
from prophet import Prophet

DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DATA_DIR, "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading Japan earthquake data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "earthquakes_japan_raw.csv"))
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    print(f"Total records: {len(df)} (mag 4.5+)")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Max magnitude: {df['mag'].max()}")

    # Model 1: Monthly earthquake count (all mag 4.5+)
    print("\nTraining model: Japan monthly earthquake count...")
    monthly = df.resample("MS", on="date").size().reset_index()
    monthly.columns = ["ds", "y"]

    model_count = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model_count.fit(monthly)

    with open(os.path.join(MODELS_DIR, "earthquake_japan_count.pkl"), "wb") as f:
        pickle.dump(model_count, f)
    print("  Saved: earthquake_japan_count.pkl")

    # Model 2: Monthly max magnitude
    print("Training model: Japan monthly max magnitude...")
    monthly_mag = df.resample("MS", on="date")["mag"].max().reset_index()
    monthly_mag.columns = ["ds", "y"]
    monthly_mag = monthly_mag.dropna()

    model_mag = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    model_mag.fit(monthly_mag)

    with open(os.path.join(MODELS_DIR, "earthquake_japan_magnitude.pkl"), "wb") as f:
        pickle.dump(model_mag, f)
    print("  Saved: earthquake_japan_magnitude.pkl")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
