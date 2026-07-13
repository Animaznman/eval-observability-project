# evaluators/judge_coverage_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import re


# Initialize observability tools
wandb = init_wandb("judge-coverage-evaluator")
bt = init_braintrust("judge-coverage-evaluator")
mlflow = init_mlflow("judge-coverage-evaluator")


def extract_bullets(text: str):
    """
    Extract bullet points from agent output.
    Supports '-', '•', '*' bullets.
    """
    pattern = r"(?:[-•*]\s+)(.*)"
    return [m.strip() for m in re.findall(pattern, text)]


def coverage_score(points: list, judge_text: str):
    """
    Compute coverage score:
    - For each point, check if the judge mentions a key phrase from it.
    - Score = (# points covered) / (# total points)
    """
    if not points:
        return 1.0, []  # nothing to cover → full score

    covered = []
    judge_lower = judge_text.lower()

    for p in points:
        # Use simple keyword matching: first 4–6 words of each bullet
        keywords = " ".join(p.lower().split()[:6])
        if keywords and keywords in judge_lower:
            covered.append(p)

    score = len(covered) / len(points)
    return score, covered


def judge_coverage_evaluator(
    ticker: str,
    date: str,
    interval: int,
    affirmative_output: str,
    negative_output: str,
    judge_output: str
):
    """
    Evaluate whether the judge:
    - Addressed each affirmative bullet point
    - Addressed each negative bullet point
    - Compared the two sides
    """

    # -----------------------------
    # Extract bullet points
    # -----------------------------
    aff_points = extract_bullets(affirmative_output)
    neg_points = extract_bullets(negative_output)

    # -----------------------------
    # Compute coverage scores
    # -----------------------------
    aff_score, aff_covered = coverage_score(aff_points, judge_output)
    neg_score, neg_covered = coverage_score(neg_points, judge_output)

    # -----------------------------
    # Comparative reasoning score
    # Judge must mention both sides
    # -----------------------------
    judge_lower = judge_output.lower()
    comparative_score = 0

    if "affirmative" in judge_lower or "bullish" in judge_lower:
        comparative_score += 0.5

    if "negative" in judge_lower or "bearish" in judge_lower:
        comparative_score += 0.5

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        # Affirmative coverage
        "affirmative_points": aff_points,
        "affirmative_covered": aff_covered,
        "affirmative_coverage_score": aff_score,

        # Negative coverage
        "negative_points": neg_points,
        "negative_covered": neg_covered,
        "negative_coverage_score": neg_score,

        # Comparative reasoning
        "comparative_reasoning_score": comparative_score,
    }

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "judge_coverage_evaluation", results)

    with start_run("judge-coverage-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_metric("affirmative_coverage_score", aff_score)
        mlflow.log_metric("negative_coverage_score", neg_score)
        mlflow.log_metric("comparative_reasoning_score", comparative_score)
        mlflow.log_text(str(results), "judge_coverage_results.txt")

    return results
