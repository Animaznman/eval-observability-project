# observability/mlflow_init.py

import mlflow

import mlflow

def init_mlflow(experiment_name):
    """
    Initialize MLflow for experiment tracking.
    This enables:
    - parameter logging
    - artifact logging
    - run metadata tracking
    - reproducible agent runs

    tracking_uri defaults to a local 'mlruns' directory,
    but can be pointed to a remote MLflow server later.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    
    return mlflow


def start_run(run_name: str = None):
    """
    Convenience wrapper for starting MLflow runs.
    Agents and orchestrators can call this instead of mlflow.start_run().
    """

    return mlflow.start_run(run_name=run_name)
