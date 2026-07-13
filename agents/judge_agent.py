# agents/judge_agent.py
# observability tools
from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Observability tools initiated
wandb = init_wandb("judge-agent")
bt = init_braintrust("judge-agent")
mlflow = init_mlflow("judge-agent")


def judge_agent(
    ticker: str,
    date: str,
    interval: int,
    affirmative_output: str,
    negative_output: str
):
    """
    Judge agent:
    - Reads both the affirmative and negative agent outputs
    - Summarizes each side's arguments
    - Compares the strength of evidence
    - Produces a verdict (not financial advice)
    - Includes an 'Observations' section listing what was evaluated
    """

    # -----------------------------
    # Step 1 — Ask the model to evaluate both sides
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Judge Analyst.\n"
                    "Your job is to evaluate the affirmative and negative arguments.\n"
                    "You must:\n"
                    "- Summarize the affirmative argument\n"
                    "- Summarize the negative argument\n"
                    "- Compare the strength of evidence\n"
                    "- Produce a verdict on which argument is stronger\n"
                    "- Include an 'Observations' section listing what you evaluated\n"
                    "Do NOT give financial advice. Only evaluate argument quality."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker}\n"
                    f"Date range: {date} to {date} + {interval} days.\n\n"
                    "Affirmative argument:\n"
                    f"{affirmative_output}\n\n"
                    "Negative argument:\n"
                    f"{negative_output}\n\n"
                    "Evaluate both sides and produce a verdict."
                ),
            },
        ],
    )

    final_output = response.choices[0].message.content

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log({
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "judge_output": final_output,
        "affirmative_input": affirmative_output,
        "negative_input": negative_output,
    })

    log_event("judge_agent_run", {
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "affirmative_input": affirmative_output,
        "negative_input": negative_output,
        "output": final_output,
    })

    with start_run("judge-agent"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_text(affirmative_output, "affirmative_input.txt")
        mlflow.log_text(negative_output, "negative_input.txt")
        mlflow.log_text(final_output, "judge_output.txt")

    return {
        "output": final_output,
        "tokens": response.usage.total_tokens
    }
