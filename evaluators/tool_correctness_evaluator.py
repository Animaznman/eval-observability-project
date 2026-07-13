# evaluators/tool_correctness_evaluator.py

from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import json


# Initialize observability tools
wandb = init_wandb("tool-correctness-evaluator")
bt = init_braintrust("tool-correctness-evaluator")
mlflow = init_mlflow("tool-correctness-evaluator")


def tool_correctness_evaluator(
    ticker: str,
    date: str,
    interval: int,
    tool_calls: list
):
    """
    Evaluate whether the agent:
    - Called the correct tools
    - Used the correct parameters
    - Used the correct date range
    - Used the correct ticker
    - Did not hallucinate tool calls
    """

    expected_tools = {
        "fetch_historical_prices",
        "fetch_headlines",
        "fetch_company_name",
    }

    # -----------------------------
    # 1. Check tool call correctness
    # -----------------------------
    called_tools = set()
    parameter_errors = []
    correct_parameter_count = 0
    total_parameter_count = 0

    for call in tool_calls:
        fn_name = call.function.name
        args = json.loads(call.function.arguments)

        called_tools.add(fn_name)

        # Count parameters
        total_parameter_count += len(args)

        # Check ticker correctness
        if "ticker" in args:
            if args["ticker"] == ticker:
                correct_parameter_count += 1
            else:
                parameter_errors.append(
                    f"Incorrect ticker: expected {ticker}, got {args['ticker']}"
                )

        # Check date correctness
        if "start_date" in args:
            if args["start_date"] == date:
                correct_parameter_count += 1
            else:
                parameter_errors.append(
                    f"Incorrect start_date: expected {date}, got {args['start_date']}"
                )

        # Check interval correctness
        if "interval" in args:
            if args["interval"] == interval:
                correct_parameter_count += 1
            else:
                parameter_errors.append(
                    f"Incorrect interval: expected {interval}, got {args['interval']}"
                )

    # -----------------------------
    # 2. Compute tool correctness score
    # -----------------------------
    missing_tools = expected_tools - called_tools
    hallucinated_tools = called_tools - expected_tools

    tool_correctness_score = (
        len(expected_tools & called_tools) / len(expected_tools)
    )

    parameter_accuracy_score = (
        correct_parameter_count / total_parameter_count
        if total_parameter_count > 0 else 0
    )

    results = {
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "expected_tools": list(expected_tools),
        "called_tools": list(called_tools),
        "missing_tools": list(missing_tools),
        "hallucinated_tools": list(hallucinated_tools),
        "tool_correctness_score": tool_correctness_score,
        "parameter_accuracy_score": parameter_accuracy_score,
        "parameter_errors": parameter_errors,
        "tool_call_count": len(tool_calls),
    }

    # -----------------------------
    # 3. Observability logging
    # -----------------------------
    wandb.log(results)

    log_event(bt, "tool_correctness_evaluation", results)

    with start_run("tool-correctness-evaluator"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_metric("tool_correctness_score", tool_correctness_score)
        mlflow.log_metric("parameter_accuracy_score", parameter_accuracy_score)
        mlflow.log_metric("tool_call_count", len(tool_calls))
        mlflow.log_text(str(results), "tool_correctness_results.txt")

    return results
