# observability/braintrust_init.py

import braintrust

def init_braintrust(project_name: str = "agentic-finance-system"):
    """
    Initialize Braintrust for evaluation and structured logging.
    This enables:
    - event logging
    - evaluator scoring (later)
    - run metadata tracking
    """

    bt = braintrust.init(
        project=project_name,
        # You can add dataset or run metadata here later
    )

    return bt


# Convenience wrapper for logging events
def log_event(event_name: str, metadata: dict):
    """
    Log an event to Braintrust.
    Used by agents to record:
    - inputs
    - outputs
    - reasoning summaries
    - tool usage summaries
    """

    braintrust.log_event(event_name, metadata)
