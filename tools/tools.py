import yfinance as yf

def fetch_historical_prices(symbol: str, start: str):
    data = yf.download(symbol, period="3mo", interval="1d", start=start)
    return data.reset_index().to_dict(orient="records")

historical_price_tool = {
    "type": "function",
    "function": {
        "name": "fetch_historical_prices",
        "description": "Fetch only 3 months of data on a daily interval from the given start time in 'YYYY-MM-DD' format.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "start": {"type": "string"}
            },
            "required": ["symbol", "start"]
        }
    }
}


import feedparser
from datetime import datetime, timedelta

def fetch_headlines(query: str, date: str, interval: int):
    """
    Fetch historical news headlines using Google News RSS.
    - query: search term
    - date: starting date (YYYY-MM-DD)
    - interval: number of days after 'date' to include
    """

    # Validate date
    try:
        start_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Date must be YYYY-MM-DD"}

    end_date = start_date + timedelta(days=interval)

    # Build Google News RSS query
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={query}+after:{start_date}+before:{end_date}"
    )

    feed = feedparser.parse(rss_url)

    results = []
    for entry in feed.entries:
        # Google News RSS returns published date in RFC822 format
        try:
            published = datetime(*entry.published_parsed[:6]).date()
        except Exception:
            continue

        if start_date <= published <= end_date:
            results.append({
                "title": entry.title,
                "url": entry.link,
                "snippet": entry.summary,
                "date": published.isoformat(),
            })

    return {
        "query": query,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "count": len(results),
        "results": results,
    }


headline_search_tool = {
    "type": "function",
    "function": {
        "name": "fetch_headlines",
        "description": "Search Google News RSS and return headlines for a given query and date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "date": {"type": "string"},
                "interval": {"type": "integer"},
            },
            "required": ["query", "date", "interval"]
        }
    }
}

import yfinance as yf

def fetch_company_name(ticker: str):
    """
    Given a stock ticker, return the associated company's name and metadata.
    Uses Yahoo Finance (free, stable).
    """

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"error": f"Unable to retrieve company info: {str(e)}"}

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "country": info.get("country"),
    }


company_lookup_tool = {
    "type": "function",
    "function": {
        "name": "fetch_company_name",
        "description": "Retrieve the company name and metadata associated with a stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"]
        }
    }
}
