""" MLflow Utils 

This module contains utility functions for MLflow.

"""
from mlflow import MlflowClient
import mlflow
from libs.configs import MLFLOW_FAKE_NEWS_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLFLOW_FAKE_NEWS_MODEL_NAME
import json
from typing import List
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_FAKE_NEWS_EXPERIMENT_NAME)

def get_artifact_uri_production() -> str:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{MLFLOW_FAKE_NEWS_MODEL_NAME}'"):
        model = dict(mv)
        if model["current_stage"] == "Production":
            production_model = model

    _run_id = production_model.get("run_id")
    return mlflow.get_run(_run_id).info.artifact_uri



def search_best_model(
        experiment_names: List[str] = [MLFLOW_FAKE_NEWS_EXPERIMENT_NAME],
        metric_name: str = "val_auc_1"
    ) -> str:
    """Search Best Run ID of given experiments
    """
    runs_  = mlflow.search_runs(experiment_names=experiment_names)
    run_id = runs_.loc[runs_[f'metrics.{metric_name}'].idxmax()]['run_id']
    artifact_path = json.loads(
        runs_[runs_["run_id"] == run_id]["tags.mlflow.log-model.history"].values[0]
    )[0]["artifact_path"]
    
    return run_id, artifact_path