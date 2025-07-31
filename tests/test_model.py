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

# Load evaluation thresholds and determine model registry name from params
PARAMS_PATH = 'params.yaml'
with open(PARAMS_PATH, 'r') as _f:
    _config = yaml.safe_load(_f)
# Model name is the DagsHub repository name
DEFAULT_MODEL = _config.get('dagshub', {}).get('repo_name')

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
        dagshub_url = _config.get('dagshub', {}).get('dagshub_url', 'https://dagshub.com')
        repo_owner = _config.get('dagshub', {}).get('repo_owner')
        repo_name = _config.get('dagshub', {}).get('repo_name')
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Load latest model version from Staging
        cls.model_name = DEFAULT_MODEL
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load vectorizer artifact
        with open('models/vectorizer.pkl', 'rb') as vf:
            cls.vectorizer = pickle.load(vf)

        # Load holdout test data
        cls.holdout_df = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name: str, stage: str = 'Staging') -> int:
        """
        Return the latest model version in the specified MLflow stage.
        Raises an error if no versions are found.
        """
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No versions found in stage '{stage}' for model '{model_name}'")
        return versions[0].version

    def test_model_loaded(self):
        """
        Assert that the MLflow model loads without error.
        """
        self.assertIsNotNone(self.model)

    def test_signature(self):
        """
        Verify that the model accepts the expected feature shape and returns correct output dimension.
        """
        sample_text = 'An amazing movie experience'
        X = self.vectorizer.transform([sample_text])
        df_input = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())

        preds = self.model.predict(df_input)
        # Input columns match vectorizer vocabulary size
        self.assertEqual(df_input.shape[1], len(self.vectorizer.get_feature_names_out()))
        # Output is 1D with one prediction
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(len(preds), df_input.shape[0])

    def test_performance(self):
        """
        Ensure model metrics on holdout data meet or exceed thresholds from params.yaml.
        """
        X_holdout = self.holdout_df.iloc[:, :-1]
        y_true = self.holdout_df.iloc[:, -1]

        y_pred = self.model.predict(X_holdout)

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Load expected thresholds
        eval_cfg = _config.get('evaluation', {}).get('thresholds', {})
        exp_acc = eval_cfg.get('accuracy', 0.4)
        exp_prec = eval_cfg.get('precision', 0.4)
        exp_rec = eval_cfg.get('recall', 0.4)
        exp_f1 = eval_cfg.get('f1', 0.4)

        self.assertGreaterEqual(acc, exp_acc, f"Accuracy {acc:.2f} < {exp_acc}")
        self.assertGreaterEqual(prec, exp_prec, f"Precision {prec:.2f} < {exp_prec}")
        self.assertGreaterEqual(rec, exp_rec, f"Recall {rec:.2f} < {exp_rec}")
        self.assertGreaterEqual(f1, exp_f1, f"F1 {f1:.2f} < {exp_f1}")


if __name__ == '__main__':
    unittest.main()
