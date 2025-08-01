#!/usr/bin/env python3
"""
Flask application for movie sentiment analysis.

Endpoints:
  /         Render the input form for movie review text.
  /predict  Accept POSTed review text, preprocess it, predict sentiment, and render result.
  /metrics  Expose Prometheus metrics for monitoring.

This app attempts to load a MLflow model from DagsHub, preferring Production stage,
falling back to Staging, and then to a local artifact if needed.
"""
import os
import time
import re
import string
import pickle
import yaml
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import mlflow
import mlflow.pyfunc
import dagshub
from mlflow.exceptions import MlflowException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Load YAML parameters
PARAMS_PATH = 'params.yaml'
with open(PARAMS_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Feature engineering method ("bow" or "tfidf")
FEATURE_METHOD = config.get('feature_engineering', {}).get('method', 'bow').lower()
# Model registry name (use DagsHub repo_name)
MODEL_NAME = config.get('dagshub', {}).get('repo_name', 'model')
# Local models directory
MODELS_DIR = Path(config.get('models_dir', 'models'))

# DagsHub & MLflow settings
dh_cfg = config.get('dagshub', {})
DAGSHUB_URL = dh_cfg.get('dagshub_url', '')
REPO_OWNER = dh_cfg.get('repo_owner', '')
REPO_NAME = dh_cfg.get('repo_name', '')
mlflow_cfg = config.get('mlflow', {})
EXPERIMENT_NAME = mlflow_cfg.get('experiment_name', '')

# ----------------------------------------------------------------------------
# MLflow / DagsHub tracking setup
# ----------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "SurajBhar"
repo_name = "moviesentiment"
# MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ----------------------------------------------------------------------------
# Inline Text Preprocessor
# ----------------------------------------------------------------------------
class TextPreprocessor:
    """
    Cleans and lemmatizes text:
      - Remove URLs, digits, punctuation
      - Lowercase and normalize whitespace
      - Remove stopwords, lemmatize tokens
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def preprocess(self, text: str) -> str:
        """
        Process a text string and return cleaned version.
        """
        if not isinstance(text, str):
            return ''
        # remove URLs
        text = self.url_pattern.sub('', text)
        # remove digits
        text = re.sub(r'\d+', '', text)
        # lowercase
        text = text.lower()
        # remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
        # whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        # tokenize and remove stopwords
        tokens = [tok for tok in text.split() if tok not in self.stop_words]
        # lemmatize
        lemmas = [self.lemmatizer.lemmatize(tok) for tok in tokens]
        return ' '.join(lemmas)

# ----------------------------------------------------------------------------
# Flask app factory
# ----------------------------------------------------------------------------
app = Flask(__name__)

# Initialize Prometheus metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    'app_request_count', 'Count of HTTP requests', ['method', 'endpoint'], registry=registry
)
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds', 'HTTP request latency', ['endpoint'], registry=registry
)
PREDICTION_COUNT = Counter(
    'model_prediction_count', 'Count of model predictions', ['prediction'], registry=registry
)

# Model loader with Production->Staging->local fallback
def get_latest_model_version(name: str):
    """
    Fetch latest model version: prefer Production, then Staging.
    """
    client = mlflow.tracking.MlflowClient()
    prod = client.get_latest_versions(name, stages=['Production'])
    if prod:
        return prod[0].version
    staging = client.get_latest_versions(name, stages=['Staging'])
    if staging:
        return staging[0].version
    return None

def load_model():
    """Load model from registry or fallback to local pickle."""
    try:
        version = get_latest_model_version(MODEL_NAME)
        if version:
            uri = f"models:/{MODEL_NAME}/{version}"
            return mlflow.pyfunc.load_model(uri)
        raise MlflowException(f"No Production/Staging versions for '{MODEL_NAME}'")
    except Exception as e:
        local_path = MODELS_DIR / 'model.pkl'
        with open(local_path, 'rb') as f:
            return pickle.load(f)

model = load_model()

# Load vectorizer
vec_file = MODELS_DIR / f"vectorizer_{FEATURE_METHOD}.pkl"
with open(vec_file, 'rb') as vf:
    vectorizer = pickle.load(vf)

preprocessor = TextPreprocessor()

@app.route('/')
def home():
    """Render input form."""
    REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
    start = time.time()
    html = render_template('index.html', result=None)
    REQUEST_LATENCY.labels(endpoint='/').observe(time.time() - start)
    return html

@app.route('/predict', methods=['POST'])
def predict():
    """Process form, predict sentiment, and render result."""
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    start = time.time()

    raw_text = request.form.get('text', '')
    cleaned = preprocessor.preprocess(raw_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start)

    return render_template('index.html', result=prediction)

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    data = generate_latest(registry)
    return data, 200, {'Content-Type': CONTENT_TYPE_LATEST}

# ----------------------------------------------------------------------------
# Entry point For local development
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker