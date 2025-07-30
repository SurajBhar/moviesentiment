import logging
import time
import re
import string
from pathlib import Path

import nltk
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Ensure necessary NLTK resources are available
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    """
    Configuration for data paths and model parameters.
    """
    DATA_PATH = Path("data.csv")
    SAMPLE_SIZE = 500
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    MAX_FEATURES = 100
    MODEL_MAX_ITER = 1000
    MLFLOW_URI = 'https://dagshub.com/SurajBhar/moviesentiment.mlflow'
    EXPERIMENT_NAME = 'Logistic Regression Baseline'


def load_and_sample_data(path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    """
    Load the dataset from CSV and return a random sample.

    Args:
        path (Path): Path to the CSV file.
        sample_size (int): Number of samples to draw.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Sampled data.
    """
    df = pd.read_csv(path)
    sampled = df.sample(sample_size, random_state=random_state)
    sampled.to_csv(path, index=False)
    logging.info(f"Loaded and sampled {sample_size} rows.")
    return sampled


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Apply normalization steps: lowercase, remove URLs, numbers, punctuation, stop words, and lemmatize.

    Args:
        text (str): Original text review.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer.
        stop_words (set): Set of stop words.

    Returns:
        str: Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'review' column in the dataset.

    Args:
        df (pd.DataFrame): DataFrame with a 'review' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned reviews.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda txt: preprocess_text(txt, lemmatizer, stop_words))
    logging.info("Completed text normalization.")
    return df


def prepare_features(df: pd.DataFrame, max_features: int):
    """
    Transform text data into features and split into train/test sets.

    Args:
        df (pd.DataFrame): DataFrame with 'review' and 'sentiment' columns.
        max_features (int): Number of features for CountVectorizer.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = df[df['sentiment'].isin(['positive', 'negative'])].copy()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    logging.info("Prepared train/test splits.")
    return X_train, X_test, y_train, y_test


def train_and_log_model(X_train, y_train, X_test, y_test):  # noqa: C901
    """
    Train a Logistic Regression model, evaluate it, and log parameters/metrics with MLflow.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    with mlflow.start_run():
        start_time = time.time()
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("max_features", Config.MAX_FEATURES)
        mlflow.log_param("test_size", Config.TEST_SIZE)
        model = LogisticRegression(max_iter=Config.MODEL_MAX_ITER)
        model.fit(X_train, y_train)
        mlflow.log_param("model_type", "LogisticRegression")
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
        logging.info(f"Evaluation metrics: {metrics}")
        mlflow.sklearn.log_model(model, "model")
        elapsed = time.time() - start_time
        logging.info(f"Run completed in {elapsed:.2f}s")


if __name__ == '__main__':
    df = load_and_sample_data(Config.DATA_PATH, Config.SAMPLE_SIZE, Config.RANDOM_STATE)
    df = normalize_dataset(df)
    X_train, X_test, y_train, y_test = prepare_features(df, Config.MAX_FEATURES)
    train_and_log_model(X_train, y_train, X_test, y_test)
