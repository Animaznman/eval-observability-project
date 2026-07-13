# evaluators/headline_grounding_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import re


# Initialize observability tools
wandb = init_wandb("headline-grounding-evaluator")
bt = init_braintrust("headline-grounding-evaluator")
mlflow = init_mlflow("headline-grounding-evaluator")


def extract_headlines(text: str):
    """
    Extract headlines from the 'Observations' section.
    Headlines are expected to appear as bullet points.
    """
    pattern = r"Observations.*?:([\s\S]*)"
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return []

    section = match.group(1)
    lines = section.split("\n")

    headlines = [
        line.strip("-• ").strip()
        for line in lines
        if line.strip()
    ]

    return headlines


def count_headline_references(text: str, headlines: list):
    """
    Count how many headlines are referenced in the main reasoning text.
    Deterministic keyword matching.
    """
    text_lower = text.lower()
    count = 0

    for hl in headlines:
        # Use first 4–6 words as reference anchor
        keywords = " ".join(hl.lower().split()[:6])
        if keywords and keywords in text_lower:
            count += 1

    return count


def grounding_score(headlines: list, references: int):
    """
    Score headline grounding:
    - If no headlines → score = 0
    - Otherwise → references / total headlines
    """
    if not headlines:
        return 0.0

    return references / len(headlines)


def headline_grounding_evaluator(
    ticker: str,
    date: str,
    interval: int,
    affirmative_output: str,
    negative_output: str,
    judge_output: str
):
    """
    Evaluate headline grounding for all three agents:
    - Number of headlines cited
    - Number of headlines referenced in reasoning
    - Grounding score (0-1)
    """

    # -----------------------------
    # Extract headlines
    # -----------------------------
    aff_headlines = extract_headlines(affirmative_output)
    neg_headlines = extract_headlines(negative_output)
    judge_headlines = extract_headlines(judge_output)

    # -----------------------------
    # Count references
    # -----------------------------
    aff_refs = count_headline_references(affirmative_output, aff_headlines)
    neg_refs = count_headline_references(negative_output, neg_headlines)
    judge_refs = count_headline_references(judge_output, judge_headlines)

    # -----------------------------
    # Compute grounding scores
    # -----------------------------
    aff_score = grounding_score(aff_headlines, aff_refs)
    neg_score = grounding_score(neg_headlines, neg_refs)
    judge_score = grounding_score(judge_headlines, judge_refs)

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        # Affirmative
        "affirmative_headlines": aff_headlines,
        "affirmative_headline_count": len(aff_headlines),
        "affirmative_references": aff_refs,
        "affirmative_grounding_score": aff_score,

        # Negative
        "negative_headlines": neg_headlines,
        "negative_headline_count": len(neg_headlines),
        "negative_references": neg_refs,
        "negative_grounding_score": neg_score,

        # Judge
        "judge_headlines": judge_headlines,
        "judge_headline_count": len(judge_headlines),
        "judge_references": judge_refs,
        "judge_grounding_score": judge_score,
    }

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log(results)

    log_event("headline_grounding_evaluation", results)

    with start_run("headline-grounding-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)

        mlflow.log_metric("affirmative_grounding_score", aff_score)
        mlflow.log_metric("negative_grounding_score", neg_score)
        mlflow.log_metric("judge_grounding_score", judge_score)

        mlflow.log_text(str(results), "headline_grounding_results.txt")

    return results
