"""
Natural Disaster Prediction Tool — Tornado & Earthquake forecasting.

Uses Prophet models trained on:
- NOAA historical tornado data (1950-2023, USA)
- USGS historical earthquake data (1950-2024, Japan)
"""

import os
import pickle
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import tool

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "disasters", "models"
)

# State name to abbreviation mapping (tornados).
STATE_NAMES = {
    "texas": "TX", "kansas": "KS", "oklahoma": "OK", "florida": "FL",
    "nebraska": "NE", "illinois": "IL", "mississippi": "MS", "alabama": "AL",
    "iowa": "IA", "missouri": "MO",
}

AVAILABLE_STATES = ["TX", "KS", "OK", "FL", "NE", "IL", "MS", "AL", "IA", "MO", "US"]


def _load_model(filename: str):
    """Load a trained Prophet model."""
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _predict_tornado(state_code: str) -> str:
    """Generate tornado forecast for a US state."""
    model = _load_model(f"tornado_{state_code}.pkl")
    if model is None:
        return f"No tornado model available for {state_code}."

    future = model.make_future_dataframe(periods=36, freq="MS")
    forecast = model.predict(future)

    # Filter to next 12 months from today.
    now = pd.Timestamp(datetime.now().replace(day=1))
    future_forecast = forecast[forecast["ds"] >= now].head(12)

    state_label = state_code if state_code != "US" else "United States (national)"
    result = f"🌪️ Tornado Risk Forecast — {state_label}\n"
    result += f"Model: Prophet | Data: NOAA 1950-2023 (70,022 records)\n\n"
    result += "Month          | Predicted Count | Risk Level\n"
    result += "---------------|----------------|-----------\n"

    for _, row in future_forecast.iterrows():
        month_str = row["ds"].strftime("%B %Y")
        predicted = max(0, round(row["yhat"], 1))
        lower = max(0, round(row["yhat_lower"], 1))
        upper = max(0, round(row["yhat_upper"], 1))

        if predicted > 20:
            risk = "🔴 HIGH"
        elif predicted > 10:
            risk = "🟠 MEDIUM"
        elif predicted > 5:
            risk = "🟡 LOW"
        else:
            risk = "🟢 MINIMAL"

        result += f"{month_str:<15}| {predicted:>5} ({lower}-{upper}) | {risk}\n"

    peak = future_forecast.loc[future_forecast["yhat"].idxmax()]
    result += f"\nPeak risk: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat']))} tornadoes)"
    return result


def _predict_earthquake_japan() -> str:
    """Generate earthquake forecast for Japan."""
    model_count = _load_model("earthquake_japan_count.pkl")
    model_mag = _load_model("earthquake_japan_magnitude.pkl")

    if model_count is None:
        return "No earthquake model available for Japan."

    # Forecast count — extend far enough to cover current year.
    future = model_count.make_future_dataframe(periods=36, freq="MS")
    forecast_count = model_count.predict(future)

    # Filter to show next 12 months from today.
    now = pd.Timestamp(datetime.now().replace(day=1))
    future_months = forecast_count[forecast_count["ds"] >= now].head(12)

    # Forecast max magnitude.
    mag_forecast = None
    if model_mag:
        future_mag = model_mag.make_future_dataframe(periods=36, freq="MS")
        mag_all = model_mag.predict(future_mag)
        mag_forecast = mag_all[mag_all["ds"] >= now].head(12)

    result = "🌍 Earthquake Risk Forecast — Japan\n"
    result += "Model: Prophet | Data: USGS 1950-2024 (17,022 events, mag 4.5+)\n\n"
    result += "Month          | Predicted Events | Expected Max Mag | Risk Level\n"
    result += "---------------|-----------------|-----------------|----------\n"

    for i, (_, row) in enumerate(future_months.iterrows()):
        month_str = row["ds"].strftime("%B %Y")
        predicted = max(0, round(row["yhat"], 1))
        lower = max(0, round(row["yhat_lower"], 1))
        upper = max(0, round(row["yhat_upper"], 1))

        max_mag = "—"
        if mag_forecast is not None and i < len(mag_forecast):
            mag_row = mag_forecast.iloc[i]
            max_mag = f"{mag_row['yhat']:.1f}"

        if predicted > 30:
            risk = "🔴 HIGH"
        elif predicted > 20:
            risk = "🟠 MEDIUM"
        elif predicted > 10:
            risk = "🟡 LOW"
        else:
            risk = "🟢 MINIMAL"

        result += f"{month_str:<15}| {predicted:>5} ({lower}-{upper}) | {max_mag:>8} | {risk}\n"

    peak = future_months.loc[future_months["yhat"].idxmax()]
    result += f"\nPeak seismic activity: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat']))} events)"
    return result


def _predict_hurricane() -> str:
    """Generate Atlantic hurricane season forecast."""
    model_count = _load_model("hurricane_atlantic.pkl")
    model_intensity = _load_model("hurricane_atlantic_intensity.pkl")

    if model_count is None:
        return "No hurricane model available."

    # Forecast.
    future = model_count.make_future_dataframe(periods=36, freq="MS")
    forecast = model_count.predict(future)

    now = pd.Timestamp(datetime.now().replace(day=1))
    future_months = forecast[forecast["ds"] >= now].head(12)

    # Intensity forecast.
    intensity_forecast = None
    if model_intensity:
        future_int = model_intensity.make_future_dataframe(periods=36, freq="MS")
        int_forecast = model_intensity.predict(future_int)
        intensity_forecast = int_forecast[int_forecast["ds"] >= now].head(12)

    result = "🌀 Hurricane Risk Forecast — Atlantic Basin\n"
    result += "Model: Prophet | Data: NOAA HURDAT2 1851-2023 (1,090 hurricane events)\n"
    result += "Affects: Florida, Gulf Coast, Caribbean, Mexico\n\n"
    result += "Month          | Predicted Hurricanes | Max Wind (kt) | Risk Level\n"
    result += "---------------|--------------------|--------------|-----------\n"

    for i, (_, row) in enumerate(future_months.iterrows()):
        month_str = row["ds"].strftime("%B %Y")
        predicted = max(0, round(row["yhat"], 1))
        lower = max(0, round(row["yhat_lower"], 1))
        upper = max(0, round(row["yhat_upper"], 1))

        wind = "—"
        if intensity_forecast is not None and i < len(intensity_forecast):
            wind_val = intensity_forecast.iloc[i]["yhat"]
            wind = f"{max(0, round(wind_val))}"

        if predicted > 3:
            risk = "🔴 HIGH"
        elif predicted > 1.5:
            risk = "🟠 MEDIUM"
        elif predicted > 0.5:
            risk = "🟡 LOW"
        else:
            risk = "🟢 OFF-SEASON"

        result += f"{month_str:<15}| {predicted:>5} ({lower}-{upper}) | {wind:>8} | {risk}\n"

    # Note about hurricane season.
    result += "\nNote: Atlantic hurricane season is June 1 – November 30."
    peak = future_months.loc[future_months["yhat"].idxmax()]
    result += f"\nPeak activity: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat'], 1))} hurricanes)"
    return result


class DisasterPredictorInput(BaseModel):
    """Input schema for the disaster predictor tool."""
    query: str = Field(
        description="A query about natural disaster risk prediction. Must be a plain string. "
                    "Examples: 'tornado risk in Texas', 'earthquake forecast Japan', "
                    "'Oklahoma tornado prediction', 'Japan earthquake risk'"
    )


@tool(args_schema=DisasterPredictorInput)
def disaster_predictor_tool(query: str) -> str:
    """
    Predict natural disaster risk using ML models trained on historical data.

    Supported disasters:
    1. TORNADOS (USA) — trained on 70,022 NOAA records (1950-2023)
       Available: Texas, Kansas, Oklahoma, Florida, Nebraska, Illinois, Mississippi, Alabama, Iowa, Missouri
    2. EARTHQUAKES (Japan) — trained on 17,022 USGS records (1950-2024, mag 4.5+)
    3. HURRICANES (Atlantic) — trained on 1,090 NOAA HURDAT2 events (1851-2023)
       Affects: Florida, Gulf Coast, Caribbean, Mexico

    Input MUST be a plain string describing what you want to predict.
    Examples:
      - "tornado risk in Oklahoma"
      - "earthquake forecast Japan"
      - "hurricane risk Florida"
      - "Atlantic hurricane season forecast"
      - "US tornado prediction"

    Do NOT pass a dictionary or object. Only a plain query string.
    """
    query_lower = query.lower()

    # Detect disaster type.
    if "earthquake" in query_lower or "seismic" in query_lower or "japan" in query_lower:
        return _predict_earthquake_japan()

    if "hurricane" in query_lower or "cyclone" in query_lower or "atlantic" in query_lower \
       or "florida" in query_lower or "caribbean" in query_lower or "gulf" in query_lower:
        return _predict_hurricane()

    # Default: tornado prediction.
    state_code = None
    for name, code in STATE_NAMES.items():
        if name in query_lower or code.lower() in query_lower:
            state_code = code
            break

    if "us" in query_lower or "national" in query_lower or "united states" in query_lower:
        state_code = "US"

    if not state_code:
        state_code = "US"

    return _predict_tornado(state_code)
