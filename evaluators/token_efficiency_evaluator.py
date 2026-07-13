# evaluators/token_efficiency_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run


# Initialize observability tools
wandb = init_wandb("token-efficiency-evaluator")
bt = init_braintrust("token-efficiency-evaluator")
mlflow = init_mlflow("token-efficiency-evaluator")


def token_efficiency_evaluator(
    ticker: str,
    date: str,
    interval: int,
    affirmative_tokens: int,
    negative_tokens: int,
    judge_tokens: int,
    judge_correct: bool
):
    """
    Evaluate token efficiency:
    - Count tokens used by each agent
    - Compute total tokens
    - Compute efficiency score:
        +total_tokens if judge_correct
        -total_tokens if judge_wrong
    """

    # -----------------------------
    # 1. Compute totals
    # -----------------------------
    total_tokens = affirmative_tokens + negative_tokens + judge_tokens

    # -----------------------------
    # 2. Efficiency score
    # -----------------------------
    # Reward correct verdicts, penalize incorrect ones
    efficiency_score = total_tokens if judge_correct else -total_tokens

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,

        # Raw token counts
        "affirmative_tokens": affirmative_tokens,
        "negative_tokens": negative_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": total_tokens,

        # Efficiency metric
        "judge_correct": judge_correct,
        "token_efficiency_score": efficiency_score,
    }

    # -----------------------------
    # 3. Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "token_efficiency_evaluation", results)

    with start_run("token-efficiency-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_metric("affirmative_tokens", affirmative_tokens)
        mlflow.log_metric("negative_tokens", negative_tokens)
        mlflow.log_metric("judge_tokens", judge_tokens)
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("token_efficiency_score", efficiency_score)
        mlflow.log_text(str(results), "token_efficiency_results.txt")

    return results
