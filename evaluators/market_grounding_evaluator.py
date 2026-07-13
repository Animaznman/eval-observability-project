# NOT CURRENTLY IMPLEMENTED
# DO NOT IMPLEMENT
# evaluators/market_grounding_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run


# Initialize observability tools
wandb = init_wandb("market-grounding-evaluator")
bt = init_braintrust("market-grounding-evaluator")
mlflow = init_mlflow("market-grounding-evaluator")


# Keywords that indicate market grounding
MARKET_KEYWORDS = [
    "trend", "price", "movement", "volatility", "momentum",
    "3-month", "quarter", "historical", "market", "performance",
    "uptrend", "downtrend", "bullish", "bearish", "support", "resistance",
    "return", "gain", "loss", "sell-off", "rally"
]


def keyword_hits(text: str, keywords: list):
    """
    Count how many market-related keywords appear in the text.
    Deterministic, no LLM judgment.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits


def date_range_used(text: str, date: str, interval: int):
    """
    Check whether the agent referenced the correct date range.
    Deterministic check for the start date and interval.
    """
    text_lower = text.lower()
    date_used = date.lower() in text_lower
    interval_used = str(interval) in text_lower
    return date_used and interval_used


def market_grounding_score(keyword_count: int, date_used: bool):
    """
    Score market grounding:
    - If no keywords → score = 0
    - Otherwise → score = min(1.0, (keyword_count / 10)) boosted by date correctness
    """
    if keyword_count == 0:
        return 0.0

    base = min(1.0, keyword_count / 10)
    if date_used:
        base += 0.1  # small bonus for correct date range

    return min(1.0, base)


def market_grounding_evaluator(
    ticker: str,
    date: str,
    interval: int,
    affirmative_output: str,
    negative_output: str,
    judge_output: str
):
    """
    Evaluate market grounding for all three agents:
    - Market keyword usage
    - Correct date range usage
    - Market grounding score (0-1)
    """

    # -----------------------------
    # Keyword hits
    # -----------------------------
    aff_hits = keyword_hits(affirmative_output, MARKET_KEYWORDS)
    neg_hits = keyword_hits(negative_output, MARKET_KEYWORDS)
    judge_hits = keyword_hits(judge_output, MARKET_KEYWORDS)

    # -----------------------------
    # Date range usage
    # -----------------------------
    aff_date_used = date_range_used(affirmative_output, date, interval)
    neg_date_used = date_range_used(negative_output, date, interval)
    judge_date_used = date_range_used(judge_output, date, interval)

    # -----------------------------
    # Grounding scores
    # -----------------------------
    aff_score = market_grounding_score(aff_hits, aff_date_used)
    neg_score = market_grounding_score(neg_hits, neg_date_used)
    judge_score = market_grounding_score(judge_hits, judge_date_used)

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        # Affirmative
        "affirmative_keyword_hits": aff_hits,
        "affirmative_date_used": aff_date_used,
        "affirmative_market_grounding_score": aff_score,

        # Negative
        "negative_keyword_hits": neg_hits,
        "negative_date_used": neg_date_used,
        "negative_market_grounding_score": neg_score,

        # Judge
        "judge_keyword_hits": judge_hits,
        "judge_date_used": judge_date_used,
        "judge_market_grounding_score": judge_score,
    }

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "market_grounding_evaluation", results)

    with start_run("market-grounding-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)

        mlflow.log_metric("affirmative_market_grounding_score", aff_score)
        mlflow.log_metric("negative_market_grounding_score", neg_score)
        mlflow.log_metric("judge_market_grounding_score", judge_score)

        mlflow.log_text(str(results), "market_grounding_results.txt")

    return results
