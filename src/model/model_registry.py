import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any

import yaml
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import dagshub
from dotenv import load_dotenv

from src.logger import get_logger

# Ignore noisy warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Load environment variables (e.g., DagsHub token)
load_dotenv()
logger = get_logger(__name__)


class ModelRegistrar:
    """
    Registers a trained MLflow model to the Model Registry, transitioning it to "Staging".

    Configuration (params.yaml):
      dagshub:
        dagshub_url: "https://dagshub.com"
        repo_owner: "SurajBhar"
        repo_name: "moviesentiment"
      mlflow:
        experiment_name: "LR_Baseline_Exp1"
      models_dir: "models"
      reports_dir: "reports"
      model_registry:
        name: "moviesentiment_model"
        stage: "Staging"
    """

    def __init__(
        self,
        params_path: str = 'params.yaml',
        reports_dir: str = './reports',
    ):
        self.logger = logger
        self.params = self._load_params(params_path)

        # Directories and files
        self.reports_dir = Path(self.params.get('reports_dir', reports_dir))
        self.models_dir = Path(self.params.get('models_dir', 'models'))
        self.report_file = self.reports_dir / 'experiment_info.json'

        # Registry settings
        reg_cfg = self.params.get('model_registry', {})
        self.model_name = reg_cfg.get('name', self.params.get('dagshub', {}).get('repo_name', 'model'))
        self.stage = reg_cfg.get('stage', 'Staging')

        # DagsHub/MLflow config
        dh_cfg = self.params.get('dagshub', {})
        self.repo_owner = dh_cfg.get('repo_owner', '')
        self.repo_name = dh_cfg.get('repo_name', '')
        self.dh_url = dh_cfg.get('dagshub_url', '')

        # Token and environment
        self.token = os.getenv('CAPSTONE_TEST')
        self.ci_cd = os.getenv('CI_CD', 'false').lower() == 'true'

        # Initialize tracking
        self.mlflow_enabled = False
        self._configure_tracking()

    def _load_params(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load params from %s: %s", path, e)
            raise

    def _configure_tracking(self) -> None:
        if not self.token:
            self.logger.warning("CAPSTONE_TEST token not set; registry disabled")
            return
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.token

        try:
            if self.ci_cd:
                uri = f"{self.dh_url}/{self.repo_owner}/{self.repo_name}.mlflow"
                mlflow.set_tracking_uri(uri)
                self.logger.info("CI/CD mode: set MLflow URI to %s", uri)
            else:
                dagshub.init(
                    repo_owner=self.repo_owner,
                    repo_name=self.repo_name,
                    mlflow=True,
                )
                self.logger.info("Initialized DagsHub for %s/%s", self.repo_owner, self.repo_name)

            # Ensure experiment exists
            exp_name = self.params.get('mlflow', {}).get('experiment_name', '')
            if exp_name:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp is None:
                    mlflow.create_experiment(exp_name)
                mlflow.set_experiment(exp_name)
                self.logger.info("MLflow experiment set to '%s'", exp_name)
            self.mlflow_enabled = True
        except MlflowException as e:
            self.mlflow_enabled = False
            self.logger.warning("MLflow setup failed: %s; registry disabled", e)

    def load_model_info(self) -> Dict[str, Any]:
        try:
            with open(self.report_file, 'r') as f:
                info = json.load(f)
            self.logger.debug("Loaded model info from %s", self.report_file)
            return info
        except Exception as e:
            self.logger.error("Failed to load model info: %s", e)
            raise

    def register(self) -> None:
        info = self.load_model_info()
        run_id = info.get('run_id')
        model_path = info.get('model_path')
        if not run_id or not model_path:
            self.logger.error("Model info missing run_id or model_path")
            raise ValueError("Invalid experiment_info.json content")

        model_uri = f"runs:/{run_id}/{model_path}"
        try:
            mv = mlflow.register_model(model_uri, self.model_name)
            client = MlflowClient()
            client.transition_model_version_stage(
                name=self.model_name,
                version=mv.version,
                stage=self.stage,
                archive_existing_versions=False,
            )
            self.logger.info(
                "Registered model '%s' version %s to stage '%s'",
                self.model_name, mv.version, self.stage
            )
        except Exception as e:
            self.logger.error("Model registration failed: %s", e)
            raise

    @staticmethod
    def main():
        registrar = ModelRegistrar()
        registrar.register()


if __name__ == '__main__':
    ModelRegistrar.main()
