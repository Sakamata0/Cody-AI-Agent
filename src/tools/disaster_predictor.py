"""
Natural Disaster Prediction Tool — Tornado, Earthquake & Hurricane forecasting.

Uses Prophet models trained on:
- NOAA historical tornado data (1950-2023, USA)
- USGS historical earthquake data (1950-2024, Japan)
- NOAA HURDAT2 hurricane data (1851-2023, Atlantic)

Supports dynamic time periods: users can ask about specific months, years,
or ranges, and the tool will forecast accordingly.
"""

import os
import re
import pickle
from datetime import datetime
from typing import Optional

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

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def _parse_time_period(query: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse a time period from the user's query.

    Returns (start_date, end_date) as Timestamps.
    Supports:
      - "July 2026", "December 2027"
      - "2027", "2028"
      - "2027-2028", "2027 to 2029"
      - "next 3 years", "next 6 months"
      - Default: next 12 months from today
    """
    query_lower = query.lower()
    now = pd.Timestamp(datetime.now().replace(day=1))

    # Pattern: "next N years"
    match = re.search(r"next\s+(\d+)\s+year", query_lower)
    if match:
        years = int(match.group(1))
        return now, now + pd.DateOffset(years=years)

    # Pattern: "next N months"
    match = re.search(r"next\s+(\d+)\s+month", query_lower)
    if match:
        months = int(match.group(1))
        return now, now + pd.DateOffset(months=months)

    # Pattern: year range "2027-2028" or "2027 to 2028"
    match = re.search(r"(20\d{2})\s*[-–to]+\s*(20\d{2})", query_lower)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-01")

    # Pattern: specific "Month Year" (e.g., "July 2026", "December 2027")
    for month_name, month_num in MONTH_NAMES.items():
        pattern = rf"\b{month_name}\s+(20\d{{2}})\b"
        match = re.search(pattern, query_lower)
        if match:
            year = int(match.group(1))
            target = pd.Timestamp(f"{year}-{month_num:02d}-01")
            # Show just that month
            return target, target

    # Pattern: single year "2027" or "2028"
    match = re.search(r"\b(20[2-9]\d)\b", query_lower)
    if match:
        year = int(match.group(1))
        # Don't match years that are part of data descriptions (1950-2023, etc.)
        if year > datetime.now().year - 1:
            return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-01")

    # Pattern: "this month" or "this year"
    if "this month" in query_lower:
        return now, now

    if "this year" in query_lower:
        year = datetime.now().year
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-01")

    # Default: next 12 months
    return now, now + pd.DateOffset(months=12)


def _calculate_periods_needed(end_date: pd.Timestamp) -> int:
    """Calculate how many months of future data we need from the model.
    
    We use a generous buffer since models have different training end dates
    (typically 2023-2024). We forecast far enough to cover any requested period.
    """
    now = pd.Timestamp(datetime.now().replace(day=1))
    months_ahead = (end_date.year - now.year) * 12 + (end_date.month - now.month) + 1
    # Add 36 months buffer to account for model training end date being in 2023/2024
    return max(48, months_ahead + 36)


def _load_model(filename: str):
    """Load a trained Prophet model."""
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _filter_forecast(forecast: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Filter forecast to the requested time period."""
    if start_date == end_date:
        # Single month
        return forecast[(forecast["ds"].dt.year == start_date.year) &
                        (forecast["ds"].dt.month == start_date.month)]
    return forecast[(forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)]


def _predict_tornado(state_code: str, query: str) -> str:
    """Generate tornado forecast for a US state for the requested period."""
    model = _load_model(f"tornado_{state_code}.pkl")
    if model is None:
        return f"No tornado model available for {state_code}."

    start_date, end_date = _parse_time_period(query)
    periods_needed = _calculate_periods_needed(end_date)

    try:
        future = model.make_future_dataframe(periods=periods_needed, freq="MS")
        forecast = model.predict(future)
    except Exception as e:
        return f"Prediction model error for {state_code}: {str(e)}. The model may need retraining."

    future_forecast = _filter_forecast(forecast, start_date, end_date)

    if future_forecast.empty:
        return f"No forecast data available for the requested period ({start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')})."

    state_label = state_code if state_code != "US" else "United States (national)"
    period_label = (
        start_date.strftime("%B %Y") if start_date == end_date
        else f"{start_date.strftime('%B %Y')} → {end_date.strftime('%B %Y')}"
    )

    result = f"🌪️ Tornado Risk Forecast — {state_label}\n"
    result += f"Period: {period_label}\n"
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

    if len(future_forecast) > 1:
        peak = future_forecast.loc[future_forecast["yhat"].idxmax()]
        result += f"\nPeak risk: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat']))} tornadoes)"

    return result


def _predict_earthquake_japan(query: str) -> str:
    """Generate earthquake forecast for Japan for the requested period."""
    model_count = _load_model("earthquake_japan_count.pkl")
    model_mag = _load_model("earthquake_japan_magnitude.pkl")

    if model_count is None:
        return "No earthquake model available for Japan."

    start_date, end_date = _parse_time_period(query)
    periods_needed = _calculate_periods_needed(end_date)

    try:
        future = model_count.make_future_dataframe(periods=periods_needed, freq="MS")
        forecast_count = model_count.predict(future)
    except Exception as e:
        return f"Earthquake prediction model error: {str(e)}. The model may need retraining."

    future_months = _filter_forecast(forecast_count, start_date, end_date)

    if future_months.empty:
        return f"No forecast data available for the requested period."

    # Forecast max magnitude.
    mag_forecast = None
    if model_mag:
        try:
            future_mag = model_mag.make_future_dataframe(periods=periods_needed, freq="MS")
            mag_all = model_mag.predict(future_mag)
            mag_forecast = _filter_forecast(mag_all, start_date, end_date)
        except Exception:
            pass

    period_label = (
        start_date.strftime("%B %Y") if start_date == end_date
        else f"{start_date.strftime('%B %Y')} → {end_date.strftime('%B %Y')}"
    )

    result = "🌍 Earthquake Risk Forecast — Japan\n"
    result += f"Period: {period_label}\n"
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

    if len(future_months) > 1:
        peak = future_months.loc[future_months["yhat"].idxmax()]
        result += f"\nPeak seismic activity: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat']))} events)"

    return result


def _predict_hurricane(query: str) -> str:
    """Generate Atlantic hurricane season forecast for the requested period."""
    model_count = _load_model("hurricane_atlantic.pkl")
    model_intensity = _load_model("hurricane_atlantic_intensity.pkl")

    if model_count is None:
        return "No hurricane model available."

    start_date, end_date = _parse_time_period(query)
    periods_needed = _calculate_periods_needed(end_date)

    try:
        future = model_count.make_future_dataframe(periods=periods_needed, freq="MS")
        forecast = model_count.predict(future)
    except Exception as e:
        return f"Hurricane prediction model error: {str(e)}. The model may need retraining."

    future_months = _filter_forecast(forecast, start_date, end_date)

    if future_months.empty:
        return f"No forecast data available for the requested period."

    # Intensity forecast.
    intensity_forecast = None
    if model_intensity:
        try:
            future_int = model_intensity.make_future_dataframe(periods=periods_needed, freq="MS")
            int_forecast = model_intensity.predict(future_int)
            intensity_forecast = _filter_forecast(int_forecast, start_date, end_date)
        except Exception:
            pass

    period_label = (
        start_date.strftime("%B %Y") if start_date == end_date
        else f"{start_date.strftime('%B %Y')} → {end_date.strftime('%B %Y')}"
    )

    result = "🌀 Hurricane Risk Forecast — Atlantic Basin\n"
    result += f"Period: {period_label}\n"
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

    result += "\nNote: Atlantic hurricane season is June 1 – November 30."
    if len(future_months) > 1:
        peak = future_months.loc[future_months["yhat"].idxmax()]
        result += f"\nPeak activity: {peak['ds'].strftime('%B %Y')} (~{max(0, round(peak['yhat'], 1))} hurricanes)"

    return result


class DisasterPredictorInput(BaseModel):
    """Input schema for the disaster predictor tool."""
    query: str = Field(
        description="A query about natural disaster risk prediction. Must be a plain string. "
                    "Include the disaster type, location, AND time period if specified. "
                    "Examples: 'tornado risk in Texas July 2026', 'earthquake forecast Japan 2027', "
                    "'Oklahoma tornado prediction 2027-2028', 'hurricane risk next 2 years', "
                    "'tornado risk in Kansas this month'"
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

    IMPORTANT: Include the time period in the query if the user specified one.
    Supported time periods:
      - Specific month: "July 2026", "December 2027"
      - Specific year: "2027", "2028"
      - Year range: "2027-2028", "2027 to 2029"
      - Relative: "next 2 years", "next 6 months", "this month"
      - Default: next 12 months if no period specified

    Input MUST be a plain string. Include disaster type + location + time period.
    Examples:
      - "tornado risk in Oklahoma July 2026"
      - "earthquake forecast Japan 2027-2028"
      - "hurricane risk next 2 years"
      - "tornado prediction Texas this month"
      - "US tornado risk 2027"

    Do NOT pass a dictionary or object. Only a plain query string.
    """
    query_lower = query.lower()

    # Detect disaster type.
    if "earthquake" in query_lower or "seismic" in query_lower or "japan" in query_lower:
        return _predict_earthquake_japan(query)

    if "hurricane" in query_lower or "cyclone" in query_lower or "atlantic" in query_lower \
       or "caribbean" in query_lower or "gulf" in query_lower:
        return _predict_hurricane(query)

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

    return _predict_tornado(state_code, query)
