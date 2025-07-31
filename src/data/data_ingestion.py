# src/data/data_ingestion.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from src.logger import get_logger
from src.connections.s3_connection import S3Operations

# Load environment variables from .env
load_dotenv()

# Global pandas option
pd.set_option('future.no_silent_downcasting', True)


class DataIngestion:
    """
    Handles the data ingestion pipeline: loading parameters, retrieving data,
    preprocessing, splitting, and saving the datasets.
    AWS credentials and bucket are retrieved from environment variables.
    """
    def __init__(
        self,
        params_path: str = 'params.yaml',
        output_dir: str = './data',
        logger=None,
    ):
        self.logger = logger or get_logger(__name__)
        self.params_path = params_path
        self.output_dir = output_dir
        self.params = self._load_params()
        ingestion_cfg = self.params.get('data_ingestion', {})
        self.test_size: float = ingestion_cfg.get('test_size', 0.2)
        # S3 key path from params
        self.s3_key: str = ingestion_cfg.get('key', 'data.csv')

    def _load_params(self) -> Dict[str, Any]:
        """Load parameters from a YAML file."""
        try:
            with open(self.params_path, 'r') as f:
                params = yaml.safe_load(f)
            self.logger.debug("Loaded parameters from %s", self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error("Params file not found: %s", self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error("YAML error in %s: %s", self.params_path, e)
            raise

    def _load_data(self) -> pd.DataFrame:
        """Retrieve the dataset from S3 using environment-stored credentials."""
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION", "us-east-1")
        bucket = os.getenv("S3_BUCKET_NAME")

        if not all([access_key, secret_key, bucket]):
            self.logger.error(
                "Missing AWS credentials or bucket name in environment variables"
            )
            raise EnvironmentError("AWS credentials or bucket name not set in environment")

        self.logger.info("Fetching data from S3 bucket: %s/key=%s", bucket, self.s3_key)
        try:
            s3 = S3Operations(
                bucket_name=bucket,
                aws_access_key=access_key,
                aws_secret_key=secret_key,
                region_name=region,
            )
            df = s3.fetch_csv(self.s3_key)
            self.logger.info("Data fetched from S3 successfully (%d rows)", len(df))
            return df
        except Exception as e:
            self.logger.exception("Error fetching data from S3: %s", e)
            raise

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and map sentiment labels."""
        try:
            self.logger.info("Starting preprocessing")
            df = df[df['sentiment'].isin(['positive', 'negative'])].copy()
            df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
            self.logger.info("Preprocessing completed: %d rows", len(df))
            return df
        except KeyError as e:
            self.logger.error("Missing expected column: %s", e)
            raise

    def _split_and_save(self, df: pd.DataFrame) -> None:
        """Split the DataFrame and save train/test CSVs."""
        try:
            train_df, test_df = train_test_split(
                df, test_size=self.test_size, random_state=42
            )
            raw_dir = os.path.join(self.output_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            train_df.to_csv(os.path.join(raw_dir, 'train.csv'), index=False)
            test_df.to_csv(os.path.join(raw_dir, 'test.csv'), index=False)
            self.logger.debug("Saved train/test to %s", raw_dir)
        except Exception as e:
            self.logger.error("Error during split/save: %s", e)
            raise

    def run(self) -> None:
        """Executes the full ingestion pipeline."""
        try:
            df = self._load_data()
            processed = self._preprocess(df)
            self._split_and_save(processed)
            self.logger.info("Data ingestion pipeline completed successfully")
        except Exception as e:
            self.logger.exception("Data ingestion failed: %s", e)
            raise


if __name__ == '__main__':
    ingestion = DataIngestion(params_path='params.yaml', output_dir='./data')
    ingestion.run()
