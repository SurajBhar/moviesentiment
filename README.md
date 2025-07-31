# Sentiment Analysis Microservice

A **scalable**, **containerized** web application and microservice for end-to-end movie sentiment analysis. Built for Kubernetes (Amazon EKS), it integrates data ingestion, experiment tracking, model management, CI/CD, and monitoring into a single, production-ready workflow.

Built as a scalable, containerized web application and microservice for movie-sentiment analysis, this solution:

* Ingests raw data from an AWS S3 bucket
* Tracks experiments, registers models and serves predictions via MLflow
* Enables remote collaboration through DagsHub and versioned datasets with DVC
* Manages source code in Git, hosted on GitHub, with GitHub Actions automating CI/CD
* Packages each component in Docker containers and deploys them to Amazon EKS (Kubernetes)
* Monitors performance and health metrics using Prometheus and Grafana

Altogether, it delivers a fully production-ready, Kubernetes-orchestrated platform for continuous development, deployment and monitoring of movie-sentiment models.


---

## Tech Stack

| Layer                  | Technology                     |
| ---------------------- | ------------------------------ |
| Data Storage           | AWS S3                         |
| Experiment Tracking, Model Registry & Model Serving    | MLflow                         |
| Collaboration, Pipeline Automation & Data Versioning    | DagsHub, DVC                   |
| Source Control         | Git, GitHub                    |
| CI/CD                  | GitHub Actions                 |
| Containerization       | Docker                         |
| Container Orchestration| Amazon EKS (Kubernetes)        |
| Monitoring & Alerts    | Prometheus, Grafana            |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup & Installation](#setup--installation)

   * [Conda Environment](#conda-environment)
   * [Environment Variables](#environment-variables)
5. [Local Development](#local-development)

   * [Data Pipeline](#data-pipeline)
   * [Model Training](#model-training)
   * [Web App](#web-app)
6. [Running Tests](#running-tests)
7. [DVC Pipeline](#dvc-pipeline)
8. [CI/CD (GitHub Actions)](#cicd-github-actions)
9. [Deployment on EKS](#deployment-on-eks)
10. [Monitoring & Dashboarding](#monitoring--dashboarding)
11. [Contributing](#contributing)
12. [License](#license)

---

## Architecture Overview

```text
+-------------+        +---------+        +--------------+
|  Data Source| --DVC->|  DVC    | --Local->| Data Pipeline|
+-------------+        +---------+        +--------------+
       |                                     |
       v                                     v
   S3 Bucket                               Features
       |                                     |
       v                                     v
 +-------------+      MLflow/DagsHub      +-----------+
 | Model Code  | <------------------------| Training  |
 +-------------+                          +-----------+
       |                                     |
       v                                     v
  model.pkl                             Predictions
       |                                     |
       v                                     v
+--------------------+    Flask App/API      +----------+
| Sentiment Analysis | <---------------------| Client UI|
+--------------------+                       +----------+
       |
       v
 Prometheus / Grafana
```

## Features

* **Data Versioning & Pipeline**: DVC stages for ingestion, preprocessing, feature engineering (BoW/TF-IDF), model training, evaluation, and registry.
* **Experiment Tracking**: MLflow + DagsHub integration with CI/CD support.
* **Modular Codebase**: OOP-patterned Python modules for each stage.
* **API / Webapp**: Flask microservice exposing `/predict` and `/metrics` (Prometheus format).
* **Monitoring & Alerts**: Custom Prometheus metrics (request count, latency, prediction distribution). Grafana dashboards.
* **CI/CD**: GitHub Actions pipeline for repro, tests, and automated model promotion.
* **Containerized & K8s-ready**: Dockerfiles for service, ready for Amazon EKS deployment.

## Prerequisites

* **Git** (≥2.30)
* **Conda** (Miniconda/Anaconda)
* **Docker**
* **AWS CLI** (configured)
* **kubectl** & **eksctl** (for EKS)

## Setup & Installation

### Conda Environment

```bash
conda create -n sentiment-scope python=3.10 -y
conda activate sentiment-scope
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in project root:

```ini
# CI/CD flag
CI_CD=false

# DagsHub / MLflow
CAPSTONE_TEST=<your-dagshub-token>

# AWS / S3
AWS_ACCESS_KEY_ID=<your-aws-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret>
AWS_REGION=us-east-1
S3_BUCKET_NAME=<your-bucket>
```

Load with:

```bash
source .env
```

---

## Local Development

### Data Pipeline

```bash
# Ensure S3 remote is configured in .dvc/config
dvc pull           # fetch raw data
dvc repro --no-update-lock   # run all stages
```

Artifacts:

* `data/raw/` (ingested CSV)
* `data/interim/` (preprocessed)
* `data/processed/` (features)
* `models/` (vectorizers, model.pkl)

### Model Training

```bash
python src/model/model_training.py
```

* Logs to MLflow + DagsHub when `CI_CD=false`.

### Web App

```bash
export FLASK_APP=flask_app/app.py
flask run
```

* Open [http://localhost:5000](http://localhost:5000)
* Metrics at `/metrics` for Prometheus scraping.

---

## Running Tests

```bash
# Model tests
pytest tests/test_model.py

# Flask app tests
pytest tests/test_flask_app.py
```

---

## DVC Pipeline

Stages defined in `dvc.yaml`:

* `data_ingestion`
* `data_preprocessing`
* `feature_engineering`
* `model_building`
* `model_evaluation`
* `model_registration`

Run full repro:

```bash
dvc repro --no-update-lock
```

---

## CI/CD (GitHub Actions)

Workflow: `.github/workflows/ci.yaml`

1. Checkout & Python setup
2. Install deps & `dvc[s3]`
3. `dvc pull` + `dvc repro`
4. Run unit & integration tests
5. Promote model to Production via `scripts/promote_model.py --ci-cd`

---

## Deployment on EKS

1. **Build & push Docker image**:

   ```bash
   docker build -t myrepo/sentiment-scope:latest .
   docker push myrepo/sentiment-scope:latest
   ```
2. **Create EKS cluster** (or use existing):

   ```bash
   eksctl create cluster --name sentiment-scope --region $AWS_REGION
   ```
3. **Apply Kubernetes manifests** (in `k8s/`):

   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

---

## Monitoring & Dashboarding

* **Prometheus** scrapes `/metrics` endpoint.
* **Grafana** dashboards visualize:

  * Request rate & latency
  * Prediction distribution
  * Resource metrics

Refer to `grafana/` for dashboard JSON definitions.

---

## Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/XYZ`
3. Make changes & tests
4. Push & open PR

---

## License

MIT © BHAR-AI Lab
