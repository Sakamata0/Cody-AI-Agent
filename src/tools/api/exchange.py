"""
Exchange Rate API Tool using Frankfurter (free, no API key required).

Fetches real-time currency exchange rates.
API docs: https://www.frankfurter.app/docs/
"""

import requests
from langchain_core.tools import tool


@tool
def exchange_rate_tool(query: str) -> str:
    """
    Get currency exchange rates or convert an amount between currencies.
    Use this tool when the user asks about exchange rates, currency conversion,
    or how much a currency is worth in another.
    Input should be in the format: "USD to EUR" or "100 USD to EUR".
    Supported currencies include: USD, EUR, GBP, JPY, TND, CAD, AUD, CHF, CNY, etc.
    """
    try:
        # Parse the query to extract amount, source, and target currency.
        parts = query.upper().replace(",", "").split()

        # Try to detect format: "100 USD to EUR" or "USD to EUR"
        amount = 1.0
        from_currency = ""
        to_currency = ""

        if parts[0].replace(".", "").isdigit():
            amount = float(parts[0])
            from_currency = parts[1]
            # "to" keyword is at index 2
            to_currency = parts[3] if len(parts) > 3 else parts[-1]
        else:
            from_currency = parts[0]
            to_currency = parts[2] if len(parts) > 2 else parts[-1]

        # Fetch exchange rate from Frankfurter API.
        url = f"https://api.frankfurter.app/latest"
        params = {"from": from_currency, "to": to_currency, "amount": amount}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "rates" not in data:
            return f"Could not find exchange rate for {from_currency} to {to_currency}."

        rate = data["rates"][to_currency]
        base_rate = rate / amount if amount != 1 else rate

        return (
            f"Exchange rate ({data['date']}):\n"
            f"  {amount} {from_currency} = {rate} {to_currency}\n"
            f"  1 {from_currency} = {base_rate:.4f} {to_currency}"
        )

    except (IndexError, ValueError):
        return (
            "Could not parse the query. Please use format: '100 USD to EUR' "
            "or 'USD to EUR'."
        )
    except requests.RequestException as e:
        return f"Error fetching exchange rates: {str(e)}"
