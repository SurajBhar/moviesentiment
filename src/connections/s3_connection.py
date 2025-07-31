# src/connections/s3_connection.py
import os
import sys
from io import StringIO
import boto3
import pandas as pd
from dotenv import load_dotenv
from src.logger import get_logger

# Load environment variables from .env
load_dotenv()

# Initialize module-level logger
logger = get_logger(__name__)


class S3Operations:
    """
    Manages S3 interactions: fetching CSV files as pandas DataFrames.
    Credentials and bucket are read from environment variables or can be overridden.
    """
    def __init__(
        self,
        aws_access_key: str = None,
        aws_secret_key: str = None,
        region_name: str = None,
        bucket_name: str = None,
    ):
        # Use provided credentials or fall back to environment
        self.aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")

        if not all([self.aws_access_key, self.aws_secret_key, self.bucket_name]):
            logger.error("Missing AWS credentials or S3 bucket name in environment variables.")
            raise EnvironmentError("AWS credentials or S3 bucket name not set.")

        # Initialize boto3 client
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region_name,
        )
        logger.info(
            "Initialized S3Operations for bucket '%s' in region '%s'",
            self.bucket_name,
            self.region_name,
        )

    def fetch_csv(self, key: str) -> pd.DataFrame:
        """
        Fetches a CSV file from the S3 bucket and returns it as a pandas DataFrame.

        Args:
            key (str): Path to the file in the S3 bucket.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            logger.info("Fetching '%s' from bucket '%s'", key, self.bucket_name)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            logger.info(
                "Successfully fetched '%s' with %d records",
                key,
                len(df),
            )
            return df
        except Exception as e:
            logger.exception(
                "Failed to fetch '%s' from S3: %s",
                key,
                e,
            )
            raise


if __name__ == "__main__":
    """
    Simple execution block to test S3Operations independently.
    Reads a test file key from S3_TEST_FILE_KEY env var (defaults to 'data.csv').
    Prints the first few rows of the fetched DataFrame.
    """
    try:
        test_key = os.getenv("S3_TEST_FILE_KEY", "data.csv")
        s3 = S3Operations()
        df = s3.fetch_csv(test_key)
        print("Fetched DataFrame preview:")
        print(df.head())
    except Exception as e:
        logger.error("S3Operations test failed: %s", e)
        sys.exit(1)
