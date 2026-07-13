# observability/wandb_init.py

import wandb

# Initialize Weights & Biases Traces
def init_wandb(project_name: str = "agentic-finance-system"):
    """
    Initialize Weights & Biases for agent observability.
    This sets up W&B Traces so you can see:
    - tool calls
    - arguments passed to tools
    - tool outputs
    - LLM messages
    - timing information
    - full agent reasoning traces
    """

    wandb.init(
        project=project_name,
        job_type="agent_run",
        config={
            "system": "multi-agent-finance",
            "observability": "wandb-traces",
        },
        # Enable W&B Traces
        settings=wandb.Settings()
    )

    return wandb
