# orchestrator/debate_runner.py

from agents.affirmative_agent import affirmative_agent
from agents.negative_agent import negative_agent
from agents.judge_agent import judge_agent

# evaluators
from evaluators.tool_correctness_evaluator import tool_correctness_evaluator
from evaluators.judge_coverage_evaluator import judge_coverage_evaluator
from evaluators.judge_accuracy_evaluator import judge_accuracy_evaluator
from evaluators.token_efficiency_evaluator import token_efficiency_evaluator
from evaluators.headline_grounding_evaluator import headline_grounding_evaluator
from evaluators.structural_quality_evaluator import structural_quality_evaluator


def run_debate_v1(ticker: str, date: str, interval: int):
    """
    Orchestrator v1:
    - Runs affirmative, negative, and judge agents
    - Calls evaluators as soon as their metrics are measurable
    - Returns a structured dictionary of all outputs + metrics

    NOTE: This assumes each agent returns:
        {
            "output": str,
            "tool_calls": list,   # raw tool_calls from OpenAI response
            "tokens": int         # total_tokens from OpenAI response
        }
    """

    # -----------------------------
    # 1. Run affirmative agent
    # -----------------------------
    aff_result = affirmative_agent(ticker, date, interval)
    affirmative_output = aff_result["output"]
    affirmative_tool_calls = aff_result.get("tool_calls", [])
    affirmative_tokens = aff_result.get("tokens", 0)

    # Tool correctness can be evaluated immediately
    aff_tool_metrics = tool_correctness_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        tool_calls=affirmative_tool_calls,
    )

    # -----------------------------
    # 2. Run negative agent
    # -----------------------------
    neg_result = negative_agent(ticker, date, interval)
    negative_output = neg_result["output"]
    negative_tool_calls = neg_result.get("tool_calls", [])
    negative_tokens = neg_result.get("tokens", 0)

    # Tool correctness for negative agent
    neg_tool_metrics = tool_correctness_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        tool_calls=negative_tool_calls,
    )

    # -----------------------------
    # 3. Run judge agent
    # -----------------------------
    judge_result = judge_agent(
        ticker=ticker,
        date=date,
        interval=interval,
        affirmative_output=affirmative_output,
        negative_output=negative_output,
    )
    judge_output = judge_result["output"]
    judge_tokens = judge_result.get("tokens", 0)

    # -----------------------------
    # 4. Judge coverage (immediately measurable)
    # -----------------------------
    coverage_metrics = judge_coverage_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        affirmative_output=affirmative_output,
        negative_output=negative_output,
        judge_output=judge_output,
    )

    # -----------------------------
    # 5. Headline grounding (immediately measurable)
    # -----------------------------
    headline_metrics = headline_grounding_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        affirmative_output=affirmative_output,
        negative_output=negative_output,
        judge_output=judge_output,
    )

    # -----------------------------
    # 6. Structural quality (immediately measurable)
    # -----------------------------
    structural_metrics = structural_quality_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        affirmative_output=affirmative_output,
        negative_output=negative_output,
        judge_output=judge_output,
    )

    # -----------------------------
    # 7. Judge accuracy (requires market data)
    # -----------------------------
    accuracy_metrics = judge_accuracy_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        judge_output=judge_output,
    )

    judge_correct = accuracy_metrics["correct"]

    # -----------------------------
    # 8. Token efficiency (requires tokens + correctness)
    # -----------------------------
    token_metrics = token_efficiency_evaluator(
        ticker=ticker,
        date=date,
        interval=interval,
        affirmative_tokens=affirmative_tokens,
        negative_tokens=negative_tokens,
        judge_tokens=judge_tokens,
        judge_correct=judge_correct,
    )

    # -----------------------------
    # 9. Aggregate results
    # -----------------------------
    return {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        "affirmative_output": affirmative_output,
        "negative_output": negative_output,
        "judge_output": judge_output,

        "affirmative_tool_metrics": aff_tool_metrics,
        "negative_tool_metrics": neg_tool_metrics,
        "coverage_metrics": coverage_metrics,
        "headline_metrics": headline_metrics,
        "structural_metrics": structural_metrics,
        "accuracy_metrics": accuracy_metrics,
        "token_metrics": token_metrics,
    }
