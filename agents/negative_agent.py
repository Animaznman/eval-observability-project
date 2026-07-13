# agents/negative_agent.py
# observability tools
from observability.wandb_init import init_wandb
from observability.braintrust_init import init_braintrust, log_event
from observability.mlflow_init import init_mlflow, start_run

import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import tool schemas + Python functions + Pydantic models
from tools.tools import (
    FetchHistoricalPricesParams,
    FetchHeadlinesParams,
    FetchCompanyNameParams,
    fetch_historical_prices,
    fetch_headlines,
    fetch_company_name,
    historical_price_tool,
    headline_search_tool,
    company_lookup_tool,
)

# Observability tools initiated
wandb = init_wandb("negative-agent")
bt = init_braintrust("negative-agent")
mlflow = init_mlflow("negative-agent")

def negative_agent(ticker: str, date: str, interval: int):
    """
    Negative agent:
    - Given a stock ticker, retrieve company metadata
    - Retrieve 3 months of historical price data
    - Retrieve relevant news headlines within a date range
    - Summarize why one might *avoid buying* the stock for a 3-month investment
    - Include explicit observations listing which headlines were used
    """

    # -----------------------------
    # Step 1 — Ask the model what tools it wants to call
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=[
            historical_price_tool,
            headline_search_tool,
            company_lookup_tool,
        ],
        tool_choice="auto",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Negative Investment Analyst.\n"
                    "Your job is to argue *why someone should NOT buy this stock for a 3-month investment horizon*.\n"
                    "You must:\n"
                    "- Retrieve company metadata\n"
                    "- Retrieve 3 months of historical price data\n"
                    "- Retrieve relevant news headlines\n"
                    "- Summarize market trends and news trends\n"
                    "- Produce bullet-point reasons to avoid buying\n"
                    "- Include an 'Observations' section listing the headlines used\n"
                    "Do NOT give financial advice. You are only summarizing signals."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker}\n"
                    f"Analyze market trends and news trends.\n"
                    f"Date range: {date} to {date} + {interval} days."
                ),
            },
        ],
    )

    # -----------------------------
    # Step 2 — Handle tool calls
    # -----------------------------
    messages = [
        {
            "role": "system",
            "content": "You are the Negative Investment Analyst.",
        },
        response.choices[0].message
    ]

    tool_calls = response.choices[0].message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            fn_name = tool_call.function.name

            try:
                raw_args = json.loads(tool_call.function.arguments)

                if fn_name == "fetch_historical_prices":
                    params = FetchHistoricalPricesParams(**raw_args)
                    tool_result = fetch_historical_prices(params)

                elif fn_name == "fetch_headlines":
                    params = FetchHeadlinesParams(**raw_args)
                    tool_result = fetch_headlines(params)

                elif fn_name == "fetch_company_name":
                    params = FetchCompanyNameParams(**raw_args)
                    tool_result = fetch_company_name(params)

                else:
                    tool_result = {"error": f"Unknown tool: {fn_name}"}

                # Only append tool message if no exception occurred
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            tool_result.model_dump() if hasattr(tool_result, "model_dump") else tool_result
                        )
                    }
                )

            except Exception as e:
                # Do NOT append a tool message
                # Instead, append a normal assistant message explaining the failure
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool call '{fn_name}' failed: {str(e)}"
                    }
                )
    else:
        # No tool calls at all — do NOT add any tool messages
        pass

    # -----------------------------
    # Step 3 — Ask the model to produce the final summary
    # -----------------------------
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "Using the tool results above, produce:\n"
                    "- A bullet-point summary of market trends\n"
                    "- A bullet-point summary of news trends\n"
                    "- A bullet-point list of reasons someone might avoid buying this stock for a 3-month horizon\n"
                    "- An 'Observations' section listing the headlines you used\n"
                    "Do NOT give financial advice. Only summarize signals."
                ),
            }
        ],
    )

    final_output = final.choices[0].message.content

    # -----------------------------
    # Observability logging
    # -----------------------------
    wandb.log({
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "negative_output": final_output
    })

    log_event("negative_agent_run", {
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "output": final_output,
    })

    with start_run("negative-agent"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("date", date)
        mlflow.log_param("interval", interval)
        mlflow.log_text(final_output, "negative_output.txt")

    return {
        "output": final_output,
        "tool_calls": response.choices[0].message.tool_calls,
        "tokens": response.usage.total_tokens
    }
