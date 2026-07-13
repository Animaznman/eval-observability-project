# observability/braintrust_init.py

import braintrust

def init_braintrust(project_name: str = "agentic-finance-system"):
    """
    Initialize Braintrust logger for structured evaluation logging.
    Returns a logger object that agents can use.
    """
    logger = braintrust.init_logger(
        project=project_name,
    )
    return logger


def log_event(logger, event_name: str, metadata: dict):
    """
    Log an event to Braintrust using the logger object.
    """
    logger.log(
        input={"event": event_name},
        output=metadata,
    )
