#!/usr/bin/env python3
"""
Promote an MLflow model version from one stage to another (e.g. Staging → Production).
Configuration is read from params.yaml and environment variables.
"""

import os
import sys
import argparse
import yaml
from dotenv import load_dotenv
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    """
    Load YAML configuration from the given path.

    Returns:
        dict: Parsed params.yaml contents.
    """
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        logger.debug("Loaded config from %s", path)
        return cfg
    except Exception as e:
        logger.error("Failed to load config '%s': %s", path, e)
        raise


class ModelPromoter:
    """
    Handles model promotion in the MLflow Registry:
       - Archives existing Production versions
       - Promotes the latest Staging version to Production
    """

    def __init__(self, model_name: str, ci_cd: bool, cfg: dict):
        """
        Args:
            model_name: Name of the registered MLflow model.
            ci_cd: If True, uses direct MLflow URI with token. Otherwise uses dagshub.init().
            cfg: The loaded params.yaml dict.
        """
        self.model_name = model_name
        self.ci_cd = ci_cd
        self.cfg = cfg

        # DagsHub / MLflow settings
        dh = cfg.get('dagshub', {})
        self.dh_url = dh.get('dagshub_url')
        self.owner = dh.get('repo_owner')
        self.repo = dh.get('repo_name')

        self.token = os.getenv("CAPSTONE_TEST")
        if not self.token:
            raise EnvironmentError("CAPSTONE_TEST environment variable must be set")

        # Authenticate MLflow
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.token

        if self.ci_cd:
            uri = f"{self.dh_url}/{self.owner}/{self.repo}.mlflow"
            mlflow.set_tracking_uri(uri)
            logger.info("CI/CD mode: MLflow URI set to %s", uri)
        else:
            try:
                import dagshub
                dagshub.init(repo_owner=self.owner, repo_name=self.repo, mlflow=True)
                logger.info("Initialized DagsHub SDK for %s/%s", self.owner, self.repo)
            except Exception as e:
                logger.warning("DagsHub.init failed: %s", e)

        self.client = MlflowClient()

    def promote(self, source_stage: str = "Staging", target_stage: str = "Production"):
        """
        Archive any existing target_stage versions and promote the newest source_stage version.
        """
        # Fetch latest staging version
        staging = self.client.get_latest_versions(self.model_name, stages=[source_stage])
        if not staging:
            raise MlflowException(f"No versions in '{source_stage}' for '{self.model_name}'")
        version = staging[0].version
        logger.info("Latest '%s' version: %s", source_stage, version)

        # Archive existing production
        prod = self.client.get_latest_versions(self.model_name, stages=[target_stage])
        for mv in prod:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=mv.version,
                stage="Archived"
            )
            logger.info("Archived '%s' version %s", self.model_name, mv.version)

        # Promote staging → production
        mv = self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=target_stage
        )
        logger.info("Promoted '%s' version %s to '%s'", self.model_name, version, target_stage)
        print(f"Model version {version} is now in '{target_stage}' stage.")


def main():
    parser = argparse.ArgumentParser(description="Promote MLflow model versions between registry stages.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--model-name", default=None, help="MLflow-registered model name")
    parser.add_argument("--ci-cd", action="store_true", help="Use CI/CD MLflow URI flow")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.params)

    # Determine model name
    model_name = args.model_name or cfg.get("dagshub", {}).get("repo_name")
    if not model_name:
        logger.error("Model name must be provided via --model-name or dagshub.repo_name in params.yaml")
        sys.exit(1)

    promoter = ModelPromoter(model_name=model_name, ci_cd=args.ci_cd, cfg=cfg)
    promoter.promote()


if __name__ == "__main__":
    main()
