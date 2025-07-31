import os
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.logger import get_logger

# Initialize module-level logger
g_logger = get_logger(__name__)


class FeatureEngineer:
    """
    Handles loading preprocessed text data, applying text vectorization (BOW or TF-IDF),
    and saving transformed datasets along with the fitted vectorizer.

    Configuration (params.yaml):
      feature_engineering:
        method: bow       # or 'tfidf'
        bow:
          max_features: 1000
        tfidf:
          max_features: 1000
    """
    def __init__(
        self,
        params_path: str = 'params.yaml',
        interim_dir: str = './data/interim',
        processed_dir: str = './data/processed',
        models_dir: str = './models',
        logger=None,
    ):
        self.logger = logger or g_logger
        self.params = self._load_params(params_path)

        fe_cfg = self.params.get('feature_engineering', {})
        # Determine method: 'bow' or 'tfidf'
        self.method = fe_cfg.get('method', 'bow').lower()
        if self.method not in ('bow', 'tfidf'):
            self.logger.error("Unsupported feature_engineering.method '%s'", self.method)
            raise ValueError(f"Unsupported feature_engineering.method '{self.method}'")

        # Load max_features for chosen method
        method_cfg = fe_cfg.get(self.method, {})
        try:
            self.max_features = int(method_cfg.get('max_features', 1000))
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid max_features for %s: %s", self.method, e)
            raise

        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)

        # Ensure directories exist
        for directory in (self.interim_dir, self.processed_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_params(self, path: str) -> Dict[str, Any]:
        """Load parameters from a YAML file."""
        try:
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            self.logger.debug("Parameters loaded from %s", path)
            return params
        except Exception as e:
            self.logger.error("Error loading params from %s: %s", path, e)
            raise

    def _load_dataframe(self, filename: str) -> pd.DataFrame:
        """Load a CSV from the interim directory and fill NaNs."""
        path = self.interim_dir / filename
        try:
            df = pd.read_csv(path)
            df.fillna('', inplace=True)
            self.logger.info("Loaded DataFrame %s with %d rows", path, len(df))
            return df
        except Exception as e:
            self.logger.error("Failed to load data from %s: %s", path, e)
            raise

    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to the processed directory as CSV."""
        path = self.processed_dir / filename
        try:
            df.to_csv(path, index=False)
            self.logger.info("Saved DataFrame to %s", path)
        except Exception as e:
            self.logger.error("Failed to save DataFrame to %s: %s", path, e)
            raise

    def _save_vectorizer(self, vectorizer, filename: str) -> None:
        """Persist the vectorizer to the models directory."""
        path = self.models_dir / filename
        try:
            with open(path, 'wb') as f:
                pickle.dump(vectorizer, f)
            self.logger.info("Saved vectorizer to %s", path)
        except Exception as e:
            self.logger.error("Failed to save vectorizer to %s: %s", path, e)
            raise

    def run(self) -> None:
        """Orchestrate the feature engineering pipeline."""
        try:
            # Load datasets
            train_df = self._load_dataframe('train_processed.csv')
            test_df = self._load_dataframe('test_processed.csv')

            X_train = train_df['review'].values
            y_train = train_df['sentiment'].values
            X_test = test_df['review'].values
            y_test = test_df['sentiment'].values

            # Initialize appropriate vectorizer
            if self.method == 'bow':
                VectorizerClass = CountVectorizer
            else:
                VectorizerClass = TfidfVectorizer

            self.logger.info(
                "Applying %s Vectorizer(max_features=%d)",
                self.method.upper(),
                self.max_features,
            )
            vectorizer = VectorizerClass(max_features=self.max_features)

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Build output DataFrames
            train_out = pd.DataFrame(X_train_vec.toarray())
            train_out['label'] = y_train
            test_out = pd.DataFrame(X_test_vec.toarray())
            test_out['label'] = y_test

            # Determine file suffixes
            suffix = 'bow' if self.method == 'bow' else 'tfidf'

            # Save transformed data and vectorizer
            self._save_dataframe(train_out, f'train_{suffix}.csv')
            self._save_dataframe(test_out, f'test_{suffix}.csv')
            self._save_vectorizer(vectorizer, f'vectorizer_{suffix}.pkl')

            self.logger.info("Feature engineering (%s) completed successfully", self.method)
        except Exception:
            self.logger.exception("Feature engineering pipeline failed")
            raise


if __name__ == '__main__':
    fe = FeatureEngineer(
        params_path='params.yaml',
        interim_dir='./data/interim',
        processed_dir='./data/processed',
        models_dir='./models'
    )
    fe.run()
