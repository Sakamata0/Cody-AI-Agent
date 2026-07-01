"""
Exchange Rate API Tool using ExchangeRate-API (free, no API key required).

Fetches real-time currency exchange rates.
Supports 160+ currencies including TND.
API: https://open.er-api.com/v6/latest/{base}
"""

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class ExchangeRateInput(BaseModel):
    """Input schema for the exchange rate tool."""
    query: str = Field(
        description="A currency conversion query string. Must be a plain string. "
                    "Format: 'AMOUNT FROM to TO' or 'FROM to TO'. "
                    "Examples: '100 USD to EUR', 'GBP to JPY', '50 EUR to TND'"
    )


@tool(args_schema=ExchangeRateInput)
def exchange_rate_tool(query: str) -> str:
    """
    Get currency exchange rates or convert an amount between currencies.
    Use this tool when the user asks about exchange rates, currency conversion,
    or how much a currency is worth in another.

    Input MUST be a plain string in one of these formats:
      - "100 USD to EUR" (converts 100 USD to EUR)
      - "USD to EUR" (gets the rate for 1 USD to EUR)
      - "50 GBP to JPY" (converts 50 GBP to JPY)
      - "500 EUR to TND" (converts 500 EUR to Tunisian Dinar)

    Do NOT pass a dictionary or object. Only a plain query string.
    Supported currencies: USD, EUR, GBP, JPY, TND, CAD, AUD, CHF, CNY, MAD, EGP, SAR, and 160+ more.
    """
    try:
        parts = query.upper().replace(",", "").split()

        amount = 1.0
        from_currency = ""
        to_currency = ""

        if parts[0].replace(".", "").isdigit():
            amount = float(parts[0])
            from_currency = parts[1]
            to_currency = parts[3] if len(parts) > 3 else parts[-1]
        else:
            from_currency = parts[0]
            to_currency = parts[2] if len(parts) > 2 else parts[-1]

        # Fetch exchange rate from ExchangeRate-API (free, no key needed).
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            return f"Could not fetch rates for {from_currency}."

        rates = data.get("rates", {})
        if to_currency not in rates:
            return f"Currency '{to_currency}' not found. Check the currency code."

        rate = rates[to_currency]
        converted = round(amount * rate, 2)

        return (
            f"Exchange rate ({data.get('time_last_update_utc', 'today')[:16]}):\n"
            f"  {amount} {from_currency} = {converted} {to_currency}\n"
            f"  1 {from_currency} = {rate} {to_currency}"
        )

    except (IndexError, ValueError):
        return (
            "Could not parse the query. Please use format: '100 USD to EUR' "
            "or 'USD to EUR'."
        )
    except requests.RequestException as e:
        return f"Error fetching exchange rates: {str(e)}"
