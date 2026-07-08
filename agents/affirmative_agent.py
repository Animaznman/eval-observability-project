# agents/affirmative_agent.py

# observability tools
from observability.weave_init import weave
from observability.braintrust_init import bt
from observability.mlflow_init import mlflow


import json
from openai import OpenAI

# Import tool schemas + Python functions
from tools import (
    historical_price_tool,
    fetch_historical_prices,
    headline_search_tool,
    fetch_headlines,
    company_lookup_tool,
    fetch_company_name,
)

client = OpenAI()

@weave.op
def affirmative_agent(ticker: str, date: str, interval: int):
    """
    Affirmative agent:
    - Given a stock ticker, retrieve company metadata
    - Retrieve 3 months of historical price data
    - Retrieve relevant news headlines within a date range
    - Summarize why one might buy the stock for a 3-month investment
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
                    "You are the Affirmative Investment Analyst.\n"
                    "Your job is to argue *why someone should buy this stock for a 3-month investment horizon*.\n"
                    "You must:\n"
                    "- Retrieve company metadata\n"
                    "- Retrieve 3 months of historical price data\n"
                    "- Retrieve relevant news headlines\n"
                    "- Summarize market trends and news trends\n"
                    "- Produce bullet-point reasons to buy\n"
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
            "content": "You are the Affirmative Investment Analyst.",
        }
    ]

    # The model may request multiple tool calls
    for tool_call in response.choices[0].message.tool_calls or []:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # Execute the correct tool
        if fn_name == "fetch_historical_prices":
            tool_result = fetch_historical_prices(**args)

        elif fn_name == "fetch_headlines":
            tool_result = fetch_headlines(**args)

        elif fn_name == "fetch_company_name":
            tool_result = fetch_company_name(**args)

        else:
            tool_result = {"error": f"Unknown tool: {fn_name}"}

        # Add tool result to messages
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            }
        )

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
                    "- A bullet-point list of reasons someone might buy this stock for a 3-month horizon\n"
                    "- An 'Observations' section listing the headlines you used\n"
                    "Do NOT give financial advice. Only summarize signals."
                ),
            }
        ],
    )

    bt.log_event(
    "affirmative_agent_run",
    metadata={
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "output": final.choices[0].message.content,
        }
    )


    return final.choices[0].message.content