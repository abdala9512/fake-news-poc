"""
NLP Utilities
"""
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import unicodedata
import string

nltk.download("punkt")
nltk.download("stopwords")


def process_text(text, keep_as_list=False):
    """
    Input:
        text: a string containing a text
    Output:
        text_clean: a list of words containing the processed text

    """
    stemmer = SnowballStemmer("spanish")
    stopwords_ = stopwords.words("spanish")
    text_tokens = word_tokenize(text)

    text_clean = []
    for word in text_tokens:
        if (
            word not in stopwords_
            and word not in string.punctuation  # remove stopwords
            and word.isalpha()
        ):  # remove punctuation
            # stemmed_word = stemmer.stem(word)
            text_clean.append(word)
    if keep_as_list:
        return text_clean
    return " ".join(text_clean)


def remove_accent(text: str):

    unaccented_text = (
        unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    )
    return unaccented_text


def basic_cleaning(dataframe: pd.DataFrame, text_cols: list = None):

    # Strip text
    for col in dataframe.select_dtypes("object").columns:
        dataframe[col] = dataframe[col].str.strip()
        dataframe[col] = dataframe[col].str.lower()

    # Remove tildes
    if text_cols:
        for col in dataframe[text_cols].columns:
            dataframe[col] = dataframe[col].apply(lambda x: remove_accent(str(x)))
    return dataframe
