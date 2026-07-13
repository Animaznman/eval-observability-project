import argparse
import json

from orchestrator.debate_runner import run_debate_v1


def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-agent stock debate system."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
    )

    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Start date for analysis (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        required=True,
        help="Number of days for the investment horizon (e.g., 90)"
    )

    args = parser.parse_args()

    print("\n=== Running Debate Orchestrator v1 ===\n")
    print(f"Ticker: {args.ticker}")
    print(f"Date: {args.date}")
    print(f"Interval: {args.interval} days\n")

    results = run_debate_v1(
        ticker=args.ticker,
        date=args.date,
        interval=args.interval
    )

    print("=== Debate Results ===\n")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
