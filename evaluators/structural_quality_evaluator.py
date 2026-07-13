# evaluators/structural_quality_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import re


# Initialize observability tools
wandb = init_wandb("structural-quality-evaluator")
bt = init_braintrust("structural-quality-evaluator")
mlflow = init_mlflow("structural-quality-evaluator")


def count_bullets(text: str):
    """
    Count bullet points in the text.
    Supports '-', '•', '*'.
    """
    return len(re.findall(r"(?:^|\n)\s*[-•*]\s+", text))


def count_sections(text: str):
    """
    Count structural sections by detecting headings.
    Headings include:
    - Market Trends
    - News Trends
    - Reasons to Buy / Reasons to Avoid
    - Observations
    - Verdict (judge)
    """
    headings = [
        "market trends",
        "news trends",
        "reasons",
        "observations",
        "verdict",
        "summary",
        "analysis",
    ]

    text_lower = text.lower()
    return sum(1 for h in headings if h in text_lower)


def length_score(text: str):
    """
    Score based on length:
    - Too short (<150 chars) → 0
    - Good length (150–1500 chars) → 1
    - Too long (>1500 chars) → 0.5
    """
    n = len(text)

    if n < 150:
        return 0.0
    if n > 1500:
        return 0.5
    return 1.0


def structural_score(bullets: int, sections: int, length_s: float):
    """
    Combine structural metrics into a single score.
    Deterministic:
    - Bullet usage: up to 0.4
    - Section usage: up to 0.4
    - Length score: up to 0.2
    """
    bullet_component = min(0.4, bullets * 0.05)  # 20 bullets → max
    section_component = min(0.4, sections * 0.1)  # 4 sections → max
    length_component = 0.2 * length_s

    return round(bullet_component + section_component + length_component, 3)


def evaluate_structure(text: str):
    """
    Evaluate structural quality of a single agent output.
    """
    bullets = count_bullets(text)
    sections = count_sections(text)
    length_s = length_score(text)
    score = structural_score(bullets, sections, length_s)

    return {
        "bullet_count": bullets,
        "section_count": sections,
        "length_score": length_s,
        "structural_quality_score": score,
    }


def structural_quality_evaluator(
    ticker: str,
    date: str,
    interval: int,
    affirmative_output: str,
    negative_output: str,
    judge_output: str
):
    """
    Evaluate structural quality for all three agents.
    """

    # -----------------------------
    # Evaluate each agent
    # -----------------------------
    aff_struct = evaluate_structure(affirmative_output)
    neg_struct = evaluate_structure(negative_output)
    judge_struct = evaluate_structure(judge_output)

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        # Affirmative
        "affirmative_bullet_count": aff_struct["bullet_count"],
        "affirmative_section_count": aff_struct["section_count"],
        "affirmative_length_score": aff_struct["length_score"],
        "affirmative_structural_quality_score": aff_struct["structural_quality_score"],

        # Negative
        "negative_bullet_count": neg_struct["bullet_count"],
        "negative_section_count": neg_struct["section_count"],
        "negative_length_score": neg_struct["length_score"],
        "negative_structural_quality_score": neg_struct["structural_quality_score"],

        # Judge
        "judge_bullet_count": judge_struct["bullet_count"],
        "judge_section_count": judge_struct["section_count"],
        "judge_length_score": judge_struct["length_score"],
        "judge_structural_quality_score": judge_struct["structural_quality_score"],
    }

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "structural_quality_evaluation", results)

    with start_run("structural-quality-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)

        mlflow.log_metric("affirmative_structural_quality_score",
                          aff_struct["structural_quality_score"])
        mlflow.log_metric("negative_structural_quality_score",
                          neg_struct["structural_quality_score"])
        mlflow.log_metric("judge_structural_quality_score",
                          judge_struct["structural_quality_score"])

        mlflow.log_text(str(results), "structural_quality_results.txt")

    return results
