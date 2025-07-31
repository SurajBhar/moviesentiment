import os
import pickle
import unittest
import yaml

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load environment variables for MLflow authentication
load_dotenv()

# Load configuration
PARAMS_PATH = 'params.yaml'
with open(PARAMS_PATH, 'r') as f:
    _config = yaml.safe_load(f)

# Determine which feature engineering method was used
FEATURE_METHOD = _config.get('feature_engineering', {}).get('method', 'bow').lower()

class TestModel(unittest.TestCase):
    """
    End-to-end tests for the sentiment analysis model:
      1. Verify model loading from MLflow Staging.
      2. Validate input/output signature.
      3. Assert performance meets thresholds.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configure MLflow tracking, load model, vectorizer, and holdout dataset.
        """
        # Ensure MLflow credentials exist
        token = os.getenv('CAPSTONE_TEST')
        if not token:
            raise EnvironmentError('CAPSTONE_TEST environment variable must be set')
        os.environ['MLFLOW_TRACKING_USERNAME'] = token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token

        # MLflow tracking URI (DagsHub endpoint)
        dagshub_cfg = _config.get('dagshub', {})
        uri = f"{dagshub_cfg.get('dagshub_url','https://dagshub.com')}/" \
              f"{dagshub_cfg.get('repo_owner')}/{dagshub_cfg.get('repo_name')}.mlflow"
        mlflow.set_tracking_uri(uri)

        # Load latest model version from Staging (fallback behavior inside helper)
        cls.model_name = dagshub_cfg.get('repo_name')
        cls.model_version = cls._get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the correct vectorizer artifact
        vec_path = f"models/vectorizer_{FEATURE_METHOD}.pkl"
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}")
        cls.vectorizer = pickle.load(open(vec_path, 'rb'))

        # Load the appropriate holdout set
        test_csv = f"data/processed/test_{FEATURE_METHOD}.csv"
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Holdout set not found at {test_csv}")
        cls.holdout_df = pd.read_csv(test_csv)

    @staticmethod
    def _get_latest_model_version(model_name: str, stage: str = 'Staging') -> int:
        """
        Return the latest model version in the specified MLflow stage.
        Raises an error if no versions are found.
        """
        client = mlflow.tracking.MlflowClient()
        # Prefer Production, then Staging
        for preferred in ['Production', stage]:
            versions = client.get_latest_versions(model_name, stages=[preferred])
            if versions:
                return versions[0].version
        raise ValueError(f"No versions found in Production or {stage} for model '{model_name}'")

    def test_model_loaded(self):
        """Assert that the MLflow model loads without error."""
        self.assertIsNotNone(self.model)

    def test_signature(self):
        """
        Verify that the model accepts the expected feature shape
        and returns correct output dimension.
        """
        sample_text = "An amazing movie experience"
        X = self.vectorizer.transform([sample_text])
        df_in = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())

        preds = self.model.predict(df_in)
        # Input columns match vectorizer vocabulary size
        self.assertEqual(df_in.shape[1], len(self.vectorizer.get_feature_names_out()))
        # Output is 1D with one prediction
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(len(preds), df_in.shape[0])

    def test_performance(self):
        """
        Ensure model metrics on holdout data meet or exceed thresholds from params.yaml.
        """
        X_hold = self.holdout_df.iloc[:, :-1]
        y_true = self.holdout_df.iloc[:, -1]

        y_pred = self.model.predict(X_hold)

        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred)
        rec   = recall_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred)

        # Load expected thresholds
        eval_cfg = _config.get('evaluation', {}).get('thresholds', {})
        exp_acc   = eval_cfg.get('accuracy', 0.4)
        exp_prec  = eval_cfg.get('precision', 0.4)
        exp_rec   = eval_cfg.get('recall', 0.4)
        exp_f1    = eval_cfg.get('f1', 0.4)

        self.assertGreaterEqual(acc,  exp_acc,  f"Accuracy {acc:.2f} < {exp_acc}")
        self.assertGreaterEqual(prec, exp_prec, f"Precision {prec:.2f} < {exp_prec}")
        self.assertGreaterEqual(rec,  exp_rec,  f"Recall {rec:.2f} < {exp_rec}")
        self.assertGreaterEqual(f1,   exp_f1,   f"F1 {f1:.2f} < {exp_f1}")

if __name__ == '__main__':
    unittest.main()
