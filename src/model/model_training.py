# src/model/model_training.py
import os
import pickle
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression
import dagshub

from src.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Loads processed features, trains a specified model, logs parameters & model to MLflow (and DagsHub),
    and persists the trained model to disk.

    Expects 'params.yaml' structure:
      model:
        logistic_regression:
          C: 1.0
          penalty: "l1"
          solver: "liblinear"
      mlflow:
        tracking_uri: "https://dagshub.com/SurajBhar/moviesentiment.mlflow"
        experiment_name: "LR_Baseline_Exp1"
      dagshub:
        repo_owner: "SurajBhar"
        repo_name: "moviesentiment"
      models_dir: "models"
    """

    def __init__(
        self,
        params_path: str = 'params.yaml',
        processed_dir: str = './data/processed',
        logger=None,
    ):
        self.logger = logger or get_logger(__name__)
        self.params = self._load_params(params_path)

        # Model hyperparameters
        mr_cfg = self.params.get('model', {}).get('logistic_regression', {})
        self.C: float = float(mr_cfg.get('C', 1.0))
        self.penalty: str = mr_cfg.get('penalty', 'l2')
        self.solver: str = mr_cfg.get('solver', 'liblinear')

        # Directories
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(self.params.get('models_dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # DagsHub initialization (sets up MLflow integration)
        dh_cfg = self.params.get('dagshub', {})
        try:
            dagshub.init(
                repo_owner=dh_cfg.get('repo_owner', ''),
                repo_name=dh_cfg.get('repo_name', ''),
                mlflow=True,
            )
            self.logger.info(
                "Initialized DagsHub integration for %s/%s",
                dh_cfg.get('repo_owner'), dh_cfg.get('repo_name')
            )
        except Exception as e:
            self.logger.warning("Failed to initialize DagsHub integration: %s", e)

        # MLflow setup
        self.mlflow_enabled = True
        ml_cfg = self.params.get('mlflow', {})
        self.tracking_uri: str = ml_cfg.get('tracking_uri')
        self.experiment_name: str = ml_cfg.get('experiment_name', 'default')
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
            self.logger.info("MLflow experiment set to '%s'", self.experiment_name)
        except MlflowException as e:
            self.mlflow_enabled = False
            self.logger.warning(
                "Could not configure MLflow experiment '%s': %s. Continuing without MLflow.",
                self.experiment_name,
                e,
            )

    def _load_params(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            self.logger.debug("Loaded params from %s", path)
            return params
        except Exception as e:
            self.logger.error("Error loading params from %s: %s", path, e)
            raise

    def _load_data(self, filename: str) -> Tuple[Any, Any]:
        path = self.processed_dir / filename
        try:
            df = pd.read_csv(path)
            self.logger.info("Loaded data %s (%d rows)", path, len(df))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y
        except Exception as e:
            self.logger.error("Error loading data from %s: %s", path, e)
            raise

    def _save_model(self, model: Any, filename: str = 'model.pkl') -> None:
        path = self.models_dir / filename
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info("Saved model to %s", path)
        except Exception as e:
            self.logger.error("Error saving model to %s: %s", path, e)
            raise

    def run(self) -> None:
        """Execute training pipeline: load data, train model, log & save."""
        if self.mlflow_enabled:
            run_context = mlflow.start_run()
        else:
            run_context = None

        try:
            self.logger.info(
                "Training with hyperparameters: C=%s, penalty=%s, solver=%s",
                self.C, self.penalty, self.solver
            )
            if self.mlflow_enabled:
                mlflow.log_param('C', self.C)
                mlflow.log_param('penalty', self.penalty)
                mlflow.log_param('solver', self.solver)

            X_train, y_train = self._load_data('train_bow.csv')

            model = LogisticRegression(C=self.C, penalty=self.penalty, solver=self.solver)
            model.fit(X_train, y_train)
            self.logger.info("Model training completed")

            if self.mlflow_enabled:
                mlflow.sklearn.log_model(model, 'model')

            self._save_model(model)

            if run_context:
                run_context.__exit__(None, None, None)
        except Exception as e:
            if run_context:
                run_context.__exit__(type(e), e, e.__traceback__)
            self.logger.exception("Model training pipeline failed: %s", e)
            raise

    @staticmethod
    def main():
        trainer = ModelTrainer()
        trainer.run()


if __name__ == '__main__':
    ModelTrainer.main()
