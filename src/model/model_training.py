# src/model/model_training.py
"""
Module to train and save a sentiment analysis model using Logistic Regression.
Follows OOP pattern: encapsulates config loading, tracking setup, data handling,
model training, MLflow logging, and artifact persistence within a ModelTrainer class.
"""
import os
import pickle
import yaml
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
from src.logger import get_logger

# ------------------------------------------------------------------------------
# Environment & Logger Initialization
# ------------------------------------------------------------------------------
load_dotenv()
logger = get_logger(__name__)

class ModelTrainer:
    """
    Encapsulates the training pipeline:
      - Loads parameters from params.yaml
      - Configures MLflow & DagsHub for CI/CD or local
      - Loads processed data
      - Trains a LogisticRegression model
      - Logs params & model to MLflow
      - Saves the trained model locally
    """

    def __init__(self, params_path: str = 'params.yaml', data_dir: str = 'data/processed'):
        # Load configuration
        self.params = self._load_params(params_path)
        # Hyperparameters
        model_cfg = self.params.get('model', {}).get('logistic_regression', {})
        self.C = float(model_cfg.get('C', 1.0))
        self.penalty = model_cfg.get('penalty', 'l2')
        self.solver = model_cfg.get('solver', 'liblinear')
        # Directories
        self.data_dir = Path(data_dir)
        self.models_dir = Path(self.params.get('models_dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Tracking setup
        self.mlflow_enabled = False
        self._setup_tracking()

    def _load_params(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                p = yaml.safe_load(f)
            logger.debug("Loaded params from %s", path)
            return p
        except Exception as e:
            logger.error("Failed to load params: %s", e)
            raise

    def _setup_tracking(self) -> None:
        # Read tracking config
        dh_cfg = self.params.get('dagshub', {})
        ml_cfg = self.params.get('mlflow', {})
        tracking_uri = ml_cfg.get('tracking_uri')
        experiment_name = ml_cfg.get('experiment_name', 'default')
        ci_cd = os.getenv('CI_CD', 'false').lower() == 'true'
        token = os.getenv('CAPSTONE_TEST')
        # CI/CD MLflow
        if ci_cd and token and tracking_uri:
            os.environ['MLFLOW_TRACKING_USERNAME'] = token
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("CI/CD MLflow URI set to %s", tracking_uri)
        else:
            # Local DagsHub integration
            try:
                import dagshub
                dagshub.init(
                    repo_owner=dh_cfg.get('repo_owner', ''),
                    repo_name=dh_cfg.get('repo_name', ''),
                    mlflow=True
                )
                logger.info("Initialized DagsHub for %s/%s",
                            dh_cfg.get('repo_owner'), dh_cfg.get('repo_name'))
            except Exception as e:
                logger.warning("DagsHub.init failed: %s", e)
        # MLflow experiment
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            self.mlflow_enabled = True
            logger.info("MLflow experiment set to '%s'", experiment_name)
        except MlflowException as e:
            logger.warning("Could not set MLflow experiment '%s': %s", experiment_name, e)

    def _load_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self.data_dir / filename
        try:
            df = pd.read_csv(path)
            logger.info("Loaded %s (%d rows)", path, len(df))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y
        except Exception as e:
            logger.error("Failed to load data %s: %s", path, e)
            raise

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        try:
            model = LogisticRegression(
                C=self.C, penalty=self.penalty, solver=self.solver
            )
            model.fit(X, y)
            logger.info("Model training completed")
            return model
        except Exception as e:
            logger.error("Training error: %s", e)
            raise

    def _save_model(self, model: Any) -> Path:
        path = self.models_dir / 'model.pkl'
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info("Saved model to %s", path)
            return path
        except Exception as e:
            logger.error("Error saving model: %s", e)
            raise

    def run(self) -> None:
        """Main pipeline: load data, train, log, and save."""
        run_ctx = mlflow.start_run() if self.mlflow_enabled else None
        try:
            # Log hyperparameters
            if self.mlflow_enabled:
                mlflow.log_params({
                    'C': self.C,
                    'penalty': self.penalty,
                    'solver': self.solver
                })
            # Data
            X, y = self._load_data('train_bow.csv')
            # Train
            model = self._train_model(X, y)
            # Log model
            if self.mlflow_enabled:
                mlflow.sklearn.log_model(model, 'model')
            # Save artifact
            self._save_model(model)
        except Exception as e:
            logger.exception("Training pipeline failed: %s", e)
            if run_ctx:
                run_ctx.__exit__(type(e), e, e.__traceback__)
            raise
        finally:
            if run_ctx:
                run_ctx.__exit__(None, None, None)

    @staticmethod
    def main():
        trainer = ModelTrainer()
        trainer.run()


if __name__ == '__main__':
    ModelTrainer.main()
