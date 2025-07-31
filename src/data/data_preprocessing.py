import os
from typing import Optional, Iterable
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import get_logger
nltk.download('wordnet')
nltk.download('stopwords')
logger = get_logger(__name__)


class TextPreprocessor:
    """
    Encapsulates text cleaning: URL removal, digit removal, lowercasing,
    punctuation stripping, stopword removal, and lemmatization.
    """
    def __init__(
        self,
        stop_words: Optional[Iterable[str]] = None,
        min_tokens: int = 3,
    ):
        self.stop_words = set(stop_words) if stop_words is not None else set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.min_tokens = min_tokens

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _tokenize(self, text: str) -> list:
        return [tok for tok in text.split() if tok not in self.stop_words]

    def _lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def preprocess(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        lemmas = self._lemmatize(tokens)
        if len(lemmas) < self.min_tokens:
            return None
        return ' '.join(lemmas)


class DataPreprocessor:
    """
    Loads CSV, applies text preprocessing to a specified column,
    and writes the cleaned DataFrame to an output directory.
    """
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        column: str = 'review',
        processor: Optional[TextPreprocessor] = None,
        logger=None,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.column = column
        self.processor = processor or TextPreprocessor()
        self.logger = logger or get_logger(__name__)

    def run(self) -> pd.DataFrame:
        try:
            self.logger.info("Loading data from %s", self.input_path)
            df = pd.read_csv(self.input_path)

            self.logger.info("Preprocessing column '%s'", self.column)
            df[self.column] = df[self.column].apply(self.processor.preprocess)
            df = df.dropna(subset=[self.column])

            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.basename(self.input_path)
            output_file = os.path.join(self.output_dir, filename.replace('.csv', '_processed.csv'))
            df.to_csv(output_file, index=False)

            self.logger.info(
                "Saved processed '%s' (%d rows) to %s",
                filename,
                len(df),
                output_file,
            )
            return df
        except Exception as e:
            self.logger.exception("Data preprocessing failed for %s: %s", self.input_path, e)
            raise


def main():
    """
    Automatically processes 'train.csv' and 'test.csv' from './data/raw/'
    and stores results in './data/interim/'.
    """
    raw_dir = os.path.join('data', 'raw')
    interim_dir = os.path.join('data', 'interim')
    os.makedirs(interim_dir, exist_ok=True)

    for split in ['train', 'test']:
        input_path = os.path.join(raw_dir, f'{split}.csv')
        preprocessor = DataPreprocessor(
            input_path=input_path,
            output_dir=interim_dir,
            column='review',
            processor=TextPreprocessor(),
            logger=logger,
        )
        preprocessor.run()


if __name__ == '__main__':
    main()
