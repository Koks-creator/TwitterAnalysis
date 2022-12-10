import re
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')


class CleanWithRegex(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name

        self.patterns = {
            "html": re.compile("<.*?>"),
            "url":  re.compile(r"http[s]?://\S+.\S+.\S+"),
            "nonascii": re.compile(r"[^\x00-\x7f][ ]?")
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for pattern in self.patterns.values():
            X[self.column_name] = X[self.column_name].apply(lambda x: pattern.sub("", x))

        return X


class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.stop_words = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self

    @staticmethod
    def remove_punct(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_stopwords(self, text):
        filtered_text = [word.lower() for word in text.split() if word.lower() not in self.stop_words]
        return " ".join(filtered_text)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.column_name] = X[self.column_name].apply(lambda x: self.remove_punct(x))
        X[self.column_name] = X[self.column_name].apply(lambda x: self.remove_stopwords(x))

        return X


class SequencesPadding(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, max_length: int, tokenizer_path: str):
        self.column_name = column_name
        self.max_length = max_length
        self.tokenizer_path = tokenizer_path

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X[self.column_name].to_numpy()

        seqs = self.tokenizer.texts_to_sequences(data)
        padded = pad_sequences(seqs, maxlen=self.max_length, padding="pre", truncating="pre")

        X["TextPadded"] = [padded[i] for i in range(len(padded))]
        return X


def setup_preprocessing_pipeline(target_column_name: str, max_length: int, tokenizer_path: str) -> Pipeline:
    """
    :param target_column_name: column name with data you want to preprocess
    :param max_length: maximum legnth for padding
    :param tokenizer_path: path to tokenizer from tensorflow.keras.preprocessing.text
    :return:
    """
    pipeline = Pipeline([
        ("clean with regex", CleanWithRegex(column_name=target_column_name)),
        ("text cleaning", CleanText(column_name=target_column_name)),
        ("padding", SequencesPadding(column_name=target_column_name, max_length=max_length, tokenizer_path=tokenizer_path))
    ])

    return pipeline
