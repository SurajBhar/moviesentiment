# 1) Builder stage: install all deps and NLTK data
FROM python:3.10-slim AS builder

WORKDIR /workspace

# Copy only requirements to leverage caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora (only once in builder)
RUN python - <<EOF
import nltk
for pkg in ("stopwords","wordnet","omw-1.4"):
    nltk.download(pkg)
EOF

# 2) Final stage: copy only what's needed at runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code
COPY app/ app/
# Copy your persisted models & vectorizer
COPY models/ models/
# Copy templates for Flask
COPY templates/ templates/
# Copy config / params if needed at runtime
COPY params.yaml .

# Expose port
EXPOSE 5000

# Ensure app/ is on PYTHONPATH so imports work
ENV PYTHONPATH=/app

# Run with Gunicorn, pointing at your factory function
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app.main:create_app()"]
