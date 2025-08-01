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
# ====== Logging Setup =====
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional


class LoggerConfigurator:
    """
    Configures and returns loggers with rotating file and console handlers.

    Attributes:
        log_dir (Path): Directory where log files are stored.
        log_file (Path): Path to the log file.
        max_bytes (int): Maximum size in bytes before rotation.
        backup_count (int): Number of backup files to keep.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        formatter (logging.Formatter): Formatter for log messages.
    """
    def __init__(
        self,
        log_dir: str = 'logs',
        log_file: Optional[str] = None,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        console_level: int = logging.INFO,
        file_level: int = logging.INFO,
        fmt: str = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    ):
        # Determine root directory
        base_dir = Path(__file__).resolve().parents[1]
        self.log_dir = base_dir / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped log file name if not provided
        if not log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{timestamp}.log"
        self.log_file = self.log_dir / log_file

        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = logging.Formatter(fmt)

    def _get_file_handler(self) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            filename=str(self.log_file),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8',
        )
        handler.setLevel(self.file_level)
        handler.setFormatter(self.formatter)
        return handler

    def _get_console_handler(self) -> logging.StreamHandler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.console_level)
        handler.setFormatter(self.formatter)
        return handler

    def configure(
        self,
        name: Optional[str] = None,
        level: int = logging.DEBUG,
    ) -> Logger:
        """
        Creates and configures a logger.

        Args:
            name (Optional[str]): Name of the logger. Root logger if None.
            level (int): Global logging level for the logger.

        Returns:
            Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Prevent duplicate handlers on re-configuration
        if not logger.handlers:
            logger.addHandler(self._get_file_handler())
            logger.addHandler(self._get_console_handler())

        return logger


# Functional helper

def get_logger(
    name: Optional[str] = None,
    log_dir: str = 'logs',
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    fmt: str = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
) -> Logger:
    """
    Helper function to quickly get a configured logger.
    """
    configurator = LoggerConfigurator(
        log_dir=log_dir,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_level=console_level,
        file_level=file_level,
        fmt=fmt,
    )
    return configurator.configure(name)
# ==========================
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
# ----------------------------------------------------------------------------
# Load configuration and initialize logger
# ----------------------------------------------------------------------------
load_dotenv()
logger = get_logger(__name__)

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

# CI/CD flag and auth token
CI_CD_MODE = os.getenv('CI_CD', 'false').lower() == 'true'
MLFLOW_TOKEN = os.getenv('CAPSTONE_TEST')

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
def setup_tracking():
    """
    Configure MLflow tracking URI and initialize DagsHub.
    Requires CI/CD mode and MLFLOW_TOKEN to set MLflow auth and URI;
    otherwise uses DagsHub SDK for local development.
    """
    if CI_CD_MODE:
        if MLFLOW_TOKEN:
            os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TOKEN
            os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TOKEN
            uri = f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow"
            mlflow.set_tracking_uri(uri)
            logger.info("CI/CD mode: MLflow auth and URI set to %s", uri)
        else:
            logger.warning(
                "CI/CD mode enabled but MLFLOW_TOKEN not set; "
                "cannot authenticate or set MLflow URI"
            )
    else:
        try:
            dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
            logger.info("Initialized DagsHub for %s/%s", REPO_OWNER, REPO_NAME)
        except Exception as e:
            logger.warning("DagsHub.init failed: %s", e)

    if EXPERIMENT_NAME:
        try:
            exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if exp is None:
                mlflow.create_experiment(EXPERIMENT_NAME)
            mlflow.set_experiment(EXPERIMENT_NAME)
            logger.info("MLflow experiment set to '%s'", EXPERIMENT_NAME)
        except Exception as e:
            logger.warning(
                "Could not set MLflow experiment '%s': %s", EXPERIMENT_NAME, e
            )

# Run tracking setup immediately
setup_tracking()

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
def create_app():
    """
    Create Flask app with routes and Prometheus metrics.
    """
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
                logger.info("Loading model %s version %s from registry", MODEL_NAME, version)
                return mlflow.pyfunc.load_model(uri)
            raise MlflowException(f"No Production/Staging versions for '{MODEL_NAME}'")
        except Exception as e:
            logger.warning("Registry load failed: %s; falling back to local model", e)
            local_path = MODELS_DIR / 'model.pkl'
            with open(local_path, 'rb') as f:
                return pickle.load(f)

    model = load_model()

    # Load vectorizer
    vec_file = MODELS_DIR / f"vectorizer_{FEATURE_METHOD}.pkl"
    with open(vec_file, 'rb') as vf:
        vectorizer = pickle.load(vf)
    logger.info("Loaded vectorizer from %s", vec_file)

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

    return app

# ----------------------------------------------------------------------------
# Entry point For local development
# ----------------------------------------------------------------------------
# if __name__ == '__main__':
#     application = create_app()
#     application.run(
#         host='0.0.0.0',
#         port=int(os.getenv('PORT', 5000)),
#         debug=not CI_CD_MODE
#     )

# For Gunicorn entrypoint (for production)
app = create_app()

