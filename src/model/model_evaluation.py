import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger import get_logger

# Load environment variables (CI_CD flag value and DagsHub token value)
load_dotenv()
logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained model on test data, logs metrics and artifacts to MLflow (via DagsHub),
    and saves local reports. Behavior switches between CI/CD and local based on CI_CD env var.

    Expects params.yaml structure:
      dagshub:
        dagshub_url: "https://dagshub.com"
        repo_owner: "SurajBhar"
        repo_name: "moviesentiment"
      mlflow:
        experiment_name: "LR_Baseline_Exp1"
      models_dir: "models"
      feature_engineering:
        method: "bow" or "tfidf"
    """

    def __init__(
        self,
        params_path: str = 'params.yaml',
        processed_dir: str = './data/processed',
        reports_dir: str = './reports',
    ):
        self.logger = logger

        # Load configuration
        self.params = self._load_params(params_path)
        # Directories
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(self.params.get('models_dir', 'models'))

        # Feature method (bow or tfidf)
        self.method = self.params.get('feature_engineering', {}).get('method', 'bow').lower()

        # DagsHub/MLflow settings
        dh_cfg = self.params.get('dagshub', {})
        self.repo_owner = dh_cfg.get('repo_owner', '')
        self.repo_name = dh_cfg.get('repo_name', '')
        self.dh_url = dh_cfg.get('dagshub_url', '')

        self.mlflow_experiment = self.params.get('mlflow', {}).get('experiment_name', 'default')
        self.ci_cd = os.getenv('CI_CD', 'false').lower() == 'true'
        self.token = os.getenv('CAPSTONE_TEST')

        # Configure MLflow and DagsHub
        self.mlflow_enabled = False
        self._configure_tracking()

    def _load_params(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.logger.debug("Loaded parameters from %s", path)
            return cfg
        except Exception as e:
            self.logger.error("Failed to load params: %s", e)
            raise

    def _configure_tracking(self) -> None:
        # Must have token for auth
        if not self.token:
            self.logger.warning("CAPSTONE_TEST token not found; tracking disabled")
            return
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.token

        if self.ci_cd:
            uri = f"{self.dh_url}/{self.repo_owner}/{self.repo_name}.mlflow"
            mlflow.set_tracking_uri(uri)
            self.logger.info("CI/CD mode: set tracking URI to %s", uri)
        else:
            try:
                dagshub.init(
                    repo_owner=self.repo_owner,
                    repo_name=self.repo_name,
                    mlflow=True,
                )
                self.logger.info("Initialized DagsHub for %s/%s", self.repo_owner, self.repo_name)
            except Exception as e:
                self.logger.warning("DagsHub init failed: %s", e)
                return

        try:
            exp = mlflow.get_experiment_by_name(self.mlflow_experiment)
            if exp is None:
                mlflow.create_experiment(self.mlflow_experiment)
            mlflow.set_experiment(self.mlflow_experiment)
            self.logger.info("MLflow experiment set to '%s'", self.mlflow_experiment)
            self.mlflow_enabled = True
        except Exception as e:
            self.logger.warning("Could not set MLflow experiment: %s", e)

    def _load_model(self, filename: str = 'model.pkl'):
        path = self.models_dir / filename
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info("Loaded model from %s", path)
            return model
        except Exception as e:
            self.logger.error("Failed to load model: %s", e)
            raise

    def _load_test_data(self) -> (Any, Any):
        filepath = self.processed_dir / f"test_{self.method}.csv"
        try:
            df = pd.read_csv(filepath)
            self.logger.info("Loaded test data %s (%d rows)", filepath, len(df))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y
        except Exception as e:
            self.logger.error("Failed to load test data: %s", e)
            raise

    def _evaluate(self, model, X, y) -> Dict[str, float]:
        try:
            preds = model.predict(X)
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y, preds),
                'precision': precision_score(y, preds, zero_division=0),
                'recall': recall_score(y, preds, zero_division=0),
                'auc': roc_auc_score(y, proba) if proba is not None else None,
            }
            self.logger.info("Computed evaluation metrics")
            return metrics
        except Exception as e:
            self.logger.error("Evaluation failed: %s", e)
            raise

    def _save_json(self, data: Dict[str, Any], filename: str) -> Path:
        path = self.reports_dir / filename
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info("Saved report to %s", path)
            return path
        except Exception as e:
            self.logger.error("Failed to save report %s: %s", filename, e)
            raise

    def run(self) -> None:
        model = self._load_model()
        X_test, y_test = self._load_test_data()
        metrics = self._evaluate(model, X_test, y_test)

        metrics_path = self._save_json(metrics, 'metrics.json')
        info = {'method': self.method}
        if self.mlflow_enabled:
            with mlflow.start_run() as run:
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                if hasattr(model, 'get_params'):
                    for k, v in model.get_params().items():
                        mlflow.log_param(k, v)
                mlflow.sklearn.log_model(model, 'model')
                mlflow.log_artifact(str(metrics_path))
                info.update({'run_id': run.info.run_id, 'model_path': 'model'})
        else:
            self.logger.info("Skipping MLflow logging")

        self._save_json(info, 'experiment_info.json')

    @staticmethod
    def main():
        evaluator = ModelEvaluator()
        evaluator.run()


if __name__ == '__main__':
    ModelEvaluator.main()
