# evaluators/judge_accuracy_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import yfinance as yf
import re
from datetime import datetime, timedelta


# Initialize observability tools
wandb = init_wandb("judge-accuracy-evaluator")
bt = init_braintrust("judge-accuracy-evaluator")
mlflow = init_mlflow("judge-accuracy-evaluator")


def extract_verdict(judge_output: str):
    """
    Extract the judge's verdict from text.
    We look for keywords indicating BUY or AVOID.
    Deterministic, no LLM judgment.
    """

    text = judge_output.lower()

    buy_keywords = ["buy", "bullish", "positive", "favorable"]
    avoid_keywords = ["avoid", "sell", "bearish", "negative", "unfavorable"]

    for kw in buy_keywords:
        if kw in text:
            return "buy"

    for kw in avoid_keywords:
        if kw in text:
            return "avoid"

    # If unclear, treat as avoid (conservative)
    return "avoid"


def compute_return(ticker: str, start_date: str, interval_days: int):
    """
    Compute the actual return over the next interval_days using yfinance.
    Returns percentage gain/loss.
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=interval_days)

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return None  # no data available

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    first_price = float(df["Close"].iloc[0])
    last_price = float(df["Close"].iloc[-1])

    return (last_price - first_price) / first_price


def judge_accuracy_evaluator(
    ticker: str,
    date: str,
    interval: int,
    judge_output: str
):
    """
    Evaluate whether the judge made the correct call.
    Computes:
    - accuracy
    - precision
    - recall
    - f1 score
    - correctness (boolean)
    - return percentage
    """

    # -----------------------------
    # 1. Extract judge verdict
    # -----------------------------
    verdict = extract_verdict(judge_output)

    # -----------------------------
    # 2. Compute actual return
    # -----------------------------
    actual_return = compute_return(ticker, date, interval)

    if actual_return is None:
        results = {
            "ticker": ticker,
            "date": date,
            "interval": interval,
            "verdict": verdict,
            "actual_return": None,
            "correct": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "note": "No market data available for evaluation."
        }

        wandb.log(results)
        log_event(bt, "judge_accuracy_evaluation", results)

        with start_run("judge-accuracy-evaluator"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("date", date)
            mlflow.log_param("interval", interval)
            mlflow.log_text(str(results), "judge_accuracy_results.txt")

        return results

    # -----------------------------
    # 3. Determine correctness
    # -----------------------------
    # If judge says BUY → correct if return > 0
    # If judge says AVOID → correct if return <= 0
    correct = (
        (verdict == "buy" and actual_return > 0) or
        (verdict == "avoid" and actual_return <= 0)
    )

    # -----------------------------
    # 4. Compute accuracy metrics
    # -----------------------------
    # Determine true label: gain or no gain
    true_gain = actual_return > 0

    # Determine predicted label: buy or avoid
    predicted_buy = (verdict == "buy")

    # Correctness
    correct = (predicted_buy and true_gain) or (not predicted_buy and not true_gain)
    accuracy = 1.0 if correct else 0.0

    # Precision:
    # correct buys / (correct buys + incorrect buys)
    if predicted_buy:
        correct_buys = 1 if true_gain else 0
        incorrect_buys = 1 if not true_gain else 0
        precision = correct_buys / (correct_buys + incorrect_buys)
    else:
        # No buy prediction → undefined precision → treat as 0
        precision = 0.0

    # Recall:
    # correct buys / (correct buys + false avoids)
    if true_gain:
        correct_buys = 1 if predicted_buy else 0
        false_avoids = 1 if not predicted_buy else 0
        recall = correct_buys / (correct_buys + false_avoids)
    else:
        # No true gain → recall undefined → treat as 0
        recall = 0.0

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "verdict": verdict,
        "actual_return": actual_return,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    # -----------------------------
    # 5. Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "judge_accuracy_evaluation", results)

    with start_run("judge-accuracy-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("actual_return", actual_return)
        mlflow.log_text(str(results), "judge_accuracy_results.txt")

    return results
