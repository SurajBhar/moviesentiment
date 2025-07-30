"""
Experiment 2: BagOfWords_vs_TFIDF_PipelineEval

This experiment systematically evaluates how two common text vectorization 
methods—Bag-of-Words (BoW) and Term-Frequency - Inverse-Document-Frequency (TF-IDF)—
impact the performance of multiple classification algorithms on 
IMDb movie reviews. 

Specifically, for each representation (BoW vs. TF-IDF), 
we train and compare five models (Logistic Regression, 
Multinomial Naive Bayes, XGBoost, Random Forest, and 
Gradient Boosting), logging all hyperparameters and metrics 
(accuracy, precision, recall, F1) via MLflow on DagsHub.

By structuring it as nested runs, we’ll be able to:
- Directly compare BoW vs. TF-IDF under identical algorithmic settings,
- Benchmark each classifier’s strengths and weaknesses against raw word-count vs. normalized term-weight features,
- Identify the best-performing pipeline (vectorizer + model) for sentiment prediction, and
- Capture all artifacts, parameters, and metrics for reproducibility and future hyperparameter tuning.

In short, this “BoW_vs_TFIDF_Comparison” experiment lays the groundwork
for selecting the optimal feature extraction technique and classifier before moving on to more advanced approaches 
(e.g., word embeddings or deep learning).
"""

import logging
import time
import re
import string
from pathlib import Path
from typing import Dict, Any

import mlflow
import mlflow.sklearn
import dagshub
import nltk
import numpy as np
import pandas as pd
import scipy.sparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Ensure required NLTK data packages are available
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    """
    Configuration for data paths, MLflow, and experiment settings.
    """
    DATA_PATH: Path = Path("notebooks/data.csv")
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    MLFLOW_URI: str = "https://dagshub.com/SurajBhar/moviesentiment.mlflow"
    REPO_OWNER: str = "SurajBhar"
    REPO_NAME: str = "moviesentiment"
    EXPERIMENT_NAME: str = "BoW vs TF-IDF"

    @classmethod
    def init_mlflow(cls) -> None:
        mlflow.set_tracking_uri(cls.MLFLOW_URI)
        dagshub.init(repo_owner=cls.REPO_OWNER, repo_name=cls.REPO_NAME, mlflow=True)
        mlflow.set_experiment(cls.EXPERIMENT_NAME)


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Clean and normalize a single text string.

    Steps:
    - Lowercase
    - Remove URLs
    - Remove digits
    - Remove punctuation
    - Remove stop words
    - Lemmatize tokens

    Args:
        text: raw text review
        lemmatizer: NLTK WordNetLemmatizer instance
        stop_words: set of English stop words

    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(lemmatizer.lemmatize(token) for token in tokens)


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """
    Load dataset, normalize text, and encode sentiments.

    Args:
        path: Path to the CSV file with 'review' and 'sentiment' columns

    Returns:
        DataFrame with cleaned 'review' and binary 'sentiment'
    """
    df = pd.read_csv(path)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df['review'] = df['review'].apply(lambda txt: preprocess_text(txt, lemmatizer, stop_words))
    df = df[df['sentiment'].isin(['positive', 'negative'])].copy()
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    logging.info(f"Loaded {len(df)} records after filtering positive/negative.")
    return df


def get_model_hyperparams(algo_name: str, model: Any) -> Dict[str, Any]:
    """
    Extract hyperparameters from model based on algorithm name.
    """
    params = {}
    if algo_name == "LogisticRegression":
        params = {"C": model.C, "max_iter": model.max_iter}
    elif algo_name == "MultinomialNB":
        params = {"alpha": model.alpha}
    elif algo_name == "XGBoost":
        params = {"n_estimators": model.n_estimators, "learning_rate": model.learning_rate}
    elif algo_name == "RandomForest":
        params = {"n_estimators": model.n_estimators, "max_depth": model.max_depth}
    elif algo_name == "GradientBoosting":
        params = {"n_estimators": model.n_estimators, "learning_rate": model.learning_rate, "max_depth": model.max_depth}
    return params


def train_and_evaluate(df: pd.DataFrame) -> None:
    """
    Run nested MLflow experiments for different vectorizers and algorithms.
    """
    Config.init_mlflow()

    vectorizers = {
        "BoW": CountVectorizer(max_features=1000),
        "TF-IDF": TfidfVectorizer(max_features=1000),
    }
    algorithms = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE),
        "MultinomialNB": MultinomialNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=Config.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=Config.RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=Config.RANDOM_STATE),
    }

    with mlflow.start_run(run_name="BoW_vs_TFIDF_Experiment"):
        for vec_name, vectorizer in vectorizers.items():
            for algo_name, algorithm in algorithms.items():
                with mlflow.start_run(run_name=f"{algo_name}_with_{vec_name}", nested=True):
                    logging.info(f"Training {algo_name} with {vec_name}")
                    X = vectorizer.fit_transform(df['review'])
                    y = df['sentiment'].values
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
                    )

                    mlflow.log_params({
                        "vectorizer": vec_name,
                        "algorithm": algo_name,
                        "test_size": Config.TEST_SIZE
                    })

                    model = algorithm
                    model.fit(X_train, y_train)

                    mlflow.log_params(get_model_hyperparams(algo_name, model))

                    y_pred = model.predict(X_test)
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred),
                        "recall": recall_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred),
                    }
                    mlflow.log_metrics(metrics)

                    example = X_test[:5].toarray() if scipy.sparse.issparse(X_test) else X_test[:5]
                    mlflow.sklearn.log_model(model, "model", input_example=example)

                    logging.info(f"Completed {algo_name} with {vec_name}: {metrics}")


if __name__ == "__main__":
    data = load_and_prepare_data(Config.DATA_PATH)
    train_and_evaluate(data)
