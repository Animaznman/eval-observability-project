import yfinance as yf
from pydantic import BaseModel, Field
from typing import List

class FetchHistoricalPricesParams(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'")
    start: str = Field(
        ...,
        description="Start date in 'YYYY-MM-DD' format from which to fetch 3 months of daily data",
    )

class FetchHeadlinesParams(BaseModel):
    query: str = Field(..., description="Search term for Google News RSS")
    date: str = Field(
        ...,
        description="Starting date in 'YYYY-MM-DD' format for the news search window",
    )
    interval: int = Field(
        ...,
        description="Number of days after 'date' to include in the search window",
    )

class FetchCompanyNameParams(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'MSFT'")

class HistoricalPrice(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class HistoricalPriceResult(BaseModel):
    prices: List[HistoricalPrice]

class Headline(BaseModel):
    title: str
    url: str
    snippet: str
    date: str   # ISO date string

class HeadlineSearchResult(BaseModel):
    query: str
    start_date: str
    end_date: str
    count: int
    results: List[Headline]

class CompanyInfo(BaseModel):
    ticker: str
    company_name: str | None
    sector: str | None
    industry: str | None
    website: str | None
    country: str | None

def fetch_historical_prices(args: FetchHistoricalPricesParams) -> HistoricalPriceResult:
    df = yf.download(
        args.symbol,
        period="3mo",
        interval="1d",
        start=args.start
    )

    # Step 1 — Reset index so Date becomes a column
    df = df.reset_index()

    # Step 2 — Flatten MultiIndex columns (e.g., ('Close','GOOG') → 'Close')
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    records = []
    for row in df.itertuples(index=False):
        records.append(
            HistoricalPrice(
                date=str(row.Date),
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=float(row.Volume),
            )
        )

    return HistoricalPriceResult(prices=records)


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

def fetch_headlines(args: FetchHeadlinesParams) -> HeadlineSearchResult:
    """
    Fetch historical news headlines using Google News RSS.
    - query: search term
    - date: starting date (YYYY-MM-DD)
    - interval: number of days after 'date' to include
    """

    # Validate date
    try:
        start_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        # Return a Pydantic model with zero results instead of a raw dict
        return HeadlineSearchResult(
            query=args.query,
            start_date=args.date,
            end_date=args.date,
            count=0,
            results=[]
        )

    end_date = start_date + timedelta(days=args.interval)

    # Build Google News RSS query
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={args.query}+after:{start_date}+before:{end_date}"
    )

    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries:
        try:
            published = datetime(*entry.published_parsed[:6]).date()
        except Exception:
            continue

        if start_date <= published <= end_date:
            headlines.append(
                Headline(
                    title=entry.title,
                    url=entry.link,
                    snippet=entry.summary,
                    date=published.isoformat(),
                )
            )

    return HeadlineSearchResult(
        query=args.query,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        count=len(headlines),
        results=headlines,
    )

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

def fetch_company_name(args: FetchCompanyNameParams) -> CompanyInfo:
    """
    Given a stock ticker, return the associated company's name and metadata.
    Uses Yahoo Finance (free, stable).
    """

    try:
        info = yf.Ticker(args.ticker).info
    except Exception as e:
        # Return a valid Pydantic model even on error
        return CompanyInfo(
            ticker=args.ticker,
            company_name=None,
            sector=None,
            industry=None,
            website=None,
            country=None,
        )

    return CompanyInfo(
        ticker=args.ticker,
        company_name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        website=info.get("website"),
        country=info.get("country"),
    )

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