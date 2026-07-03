"""
Train a Prophet model to predict monthly Atlantic hurricane activity.

Uses NOAA HURDAT2 data (1851-2023).
"""

import os
import pickle
import re
from collections import defaultdict

import pandas as pd
from prophet import Prophet

DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DATA_DIR, "models")


def parse_hurdat2(filepath: str) -> pd.DataFrame:
    """
    Parse HURDAT2 format into a DataFrame of storms with date and max wind.
    Each header line starts with a storm ID (e.g., AL012023).
    """
    storms = []
    current_storm = None

    with open(filepath, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            # Header line: storm ID, name, number of entries
            if len(parts) >= 4 and parts[0].startswith("AL"):
                current_storm = {"id": parts[0], "name": parts[1]}
                continue
            # Data line: date, time, record_id, status, lat, lon, max_wind, ...
            if len(parts) >= 7 and current_storm:
                try:
                    date_str = parts[0]  # YYYYMMDD
                    max_wind = int(parts[6]) if parts[6].strip() else 0
                    year = int(date_str[:4])
                    month = int(date_str[4:6])

                    # Only count as hurricane if max wind >= 64 knots
                    if max_wind >= 64:
                        storms.append({
                            "year": year,
                            "month": month,
                            "max_wind": max_wind,
                            "storm_id": current_storm["id"],
                        })
                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(storms)
    # Deduplicate: one entry per storm per month (take max wind)
    if not df.empty:
        df = df.groupby(["storm_id", "year", "month"]).agg({"max_wind": "max"}).reset_index()
    return df


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Parsing HURDAT2 data...")
    filepath = os.path.join(DATA_DIR, "hurricanes_atlantic_raw.txt")
    df = parse_hurdat2(filepath)
    print(f"Hurricane entries: {len(df)} (storms with wind >= 64 knots)")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # Aggregate: monthly hurricane count.
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    monthly = df.groupby(pd.Grouper(key="date", freq="MS")).size().reset_index()
    monthly.columns = ["ds", "y"]

    # Fill missing months with 0 (months with no hurricanes)
    full_range = pd.date_range(start=monthly["ds"].min(), end=monthly["ds"].max(), freq="MS")
    monthly = monthly.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    monthly.columns = ["ds", "y"]

    print(f"Monthly records: {len(monthly)}")
    print(f"Average hurricanes per month: {monthly['y'].mean():.2f}")

    # Train Prophet model.
    print("Training hurricane count model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(monthly)

    model_path = os.path.join(MODELS_DIR, "hurricane_atlantic.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")

    # Also train max intensity model.
    print("Training hurricane intensity model...")
    monthly_intensity = df.groupby(pd.Grouper(key="date", freq="MS"))["max_wind"].max().reset_index()
    monthly_intensity.columns = ["ds", "y"]
    monthly_intensity = monthly_intensity.dropna()

    model_intensity = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model_intensity.fit(monthly_intensity)

    intensity_path = os.path.join(MODELS_DIR, "hurricane_atlantic_intensity.pkl")
    with open(intensity_path, "wb") as f:
        pickle.dump(model_intensity, f)
    print(f"  Saved: {intensity_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
