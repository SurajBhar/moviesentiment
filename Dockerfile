FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/
COPY params.yaml /app/params.yaml

COPY models/vectorizer_bow.pkl /app/models/vectorizer_bow.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
# CMD ["python", "app.py"]  

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
