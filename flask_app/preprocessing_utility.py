import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure stopwords and lemmatizer are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load stop words once globally
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applies multiple text preprocessing steps:
    - Lowercasing
    - Removing stop words
    - Removing numbers
    - Removing punctuation
    - Removing URLs
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # Lowercase and tokenize
    words = text.lower().split()

    # Remove stop words, numbers, and lemmatize
    words = [
        LEMMATIZER.lemmatize(re.sub(r'\d+', '', word))  # Remove numbers and lemmatize
        for word in words if word not in STOP_WORDS
    ]

    # Remove punctuation
    cleaned_text = ' '.join(words)
    cleaned_text = re.sub(f"[{re.escape(string.punctuation)}]", " ", cleaned_text)
    cleaned_text = re.sub(r"https?://\S+|www\.\S+", "", cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Remove extra spaces

    return cleaned_text

def remove_small_sentences(df, column='text', min_words=3):
    """
    Removes rows where the specified column contains sentences with fewer than `min_words` words.
    """
    return df[df[column].apply(lambda x: len(str(x).split()) >= min_words)].reset_index(drop=True)
