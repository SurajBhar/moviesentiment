"""
# Experiment 3 : LR_Hyperparameter_GridSearch
From experiment 2:

** Second Best Algorithm**
- Logistic Regression 
**Second Best Vectoriser**
- TF-IDF

Upnext, we need to find best hyperparameters for this combination, which we are doing in experiment 3.

This experiment performs a comprehensive grid search over Logistic Regression hyperparameters using a TF–IDF text representation on IMDb movie reviews. We:

1. **Preprocess Text:**

   * Lowercase, remove digits, punctuation, and URLs
   * Filter out English stop words
   * Lemmatize each token

2. **Vectorize Reviews:**

   * Convert cleaned text into TF–IDF feature vectors

3. **Train & Tune:**

   * Use `GridSearchCV` (5-fold) to sweep over `C` (0.1, 1, 10) and penalty (`l1`, `l2`) with the `liblinear` solver
   * For each candidate, launch a **nested MLflow run**, logging the hyperparameters, cross-validation mean/std F1 scores, and final test metrics (accuracy, precision, recall, F1)
   * Identify and log the best hyperparameter set and associated model artifact

4. **Logging & Reproducibility:**

   * All runs are tracked under a top-level MLflow experiment named **“LR\_Hyperparameter\_GridSearch”** in our DagsHub repository.
   * Enables systematic comparison of hyperparameter settings and ensures the best model can be reproduced and deployed.


"""

import logging
import re
import string
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import dagshub
import nltk
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data packages are available
for pkg in ("stopwords", "wordnet", "omw-1.4"):  
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    """
    Configuration for the Logistic Regression hyperparameter tuning experiment.
    """
    DATA_PATH: Path = Path("notebooks/data.csv")
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # MLflow / DagsHub settings
    MLFLOW_URI: str = "https://dagshub.com/SurajBhar/moviesentiment.mlflow"
    REPO_OWNER: str = "SurajBhar"
    REPO_NAME: str = "moviesentiment"
    EXPERIMENT_NAME: str = "Hyperparameter_GridSearch_LR"

    # Grid search parameters
    PARAM_GRID: Dict[str, Any] = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    @classmethod
    def init_mlflow(cls) -> None:
        """
        Initialize MLflow tracking on the specified DagsHub repository,
        creating the experiment if it was deleted.
        """
        mlflow.set_tracking_uri(cls.MLFLOW_URI)
        dagshub.init(repo_owner=cls.REPO_OWNER, repo_name=cls.REPO_NAME, mlflow=True)
        try:
            mlflow.set_experiment(cls.EXPERIMENT_NAME)
        except MlflowException as e:
            # If the experiment was previously deleted, recreate it
            if "deleted" in str(e).lower() or "does not exist" in str(e).lower():
                experiment_id = mlflow.create_experiment(cls.EXPERIMENT_NAME)
                mlflow.set_experiment(experiment_id)
            else:
                raise


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Normalize input text by lowercasing, removing digits, punctuation, URLs,
    filtering stop words, and lemmatizing tokens.

    Args:
        text: Raw text input.
        lemmatizer: NLTK WordNetLemmatizer instance.
        stop_words: Set of English stop words.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens).strip()


def load_and_vectorize(
    path: Path
) -> Tuple:
    """
    Load the CSV, preprocess reviews, filter sentiment, vectorize via TF-IDF,
    and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    df = pd.read_csv(path)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df["review"] = df["review"].astype(str).apply(
        lambda txt: preprocess_text(txt, lemmatizer, stop_words)
    )
    df = df[df["sentiment"].isin(["positive", "negative"])].copy()
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, vectorizer


def train_and_tune(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer: TfidfVectorizer
) -> None:
    """
    Perform grid search on Logistic Regression hyperparameters, log runs, and save best model.
    """
    # Initialize MLflow/DagsHub
    Config.init_mlflow()

    # Start top-level run
    with mlflow.start_run(run_name="GridSearchCV_Tuning"):
        # Log global parameters inside run
        mlflow.log_params({
            "test_size": Config.TEST_SIZE,
            "random_state": Config.RANDOM_STATE,
            "param_grid": Config.PARAM_GRID,
        })

        # Execute grid search
        grid = GridSearchCV(
            LogisticRegression(),
            param_grid=Config.PARAM_GRID,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        # Nested runs for each candidate
        for params, mean, std in zip(
            grid.cv_results_["params"],
            grid.cv_results_["mean_test_score"],
            grid.cv_results_["std_test_score"],
        ):
            with mlflow.start_run(run_name=f"LR_params={params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "cv_mean_f1": mean,
                    "cv_std_f1": std,
                }
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                logging.info(f"Params={params} Metrics={metrics}")

        # Log best model after grid search
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        best_score = grid.best_score_

        mlflow.log_params({"best_params": str(best_params)})
        mlflow.log_metric("best_cv_f1", best_score)
        example = X_test[:5].toarray() if hasattr(X_test, "toarray") else X_test[:5]
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            input_example=example,
        )
        logging.info(f"Best Params={best_params} Best CV F1={best_score:.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = load_and_vectorize(Config.DATA_PATH)
    train_and_tune(X_train, X_test, y_train, y_test, vectorizer)
