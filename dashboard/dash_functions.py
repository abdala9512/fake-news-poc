import openai
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from typing import Any, Dict, List
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import mlflow

import sys
import os 

sys.path.append("./src")
from libs.configs import (MLFLOW_FAKE_NEWS_MODEL_NAME)
from libs.mlflow_utils import get_artifact_uri_production

BEST_PARAMS_ARTIFACT_PATH = f"{get_artifact_uri_production()}/best_params.json"
BEST_PARAMS = mlflow.artifacts.load_dict(BEST_PARAMS_ARTIFACT_PATH)

def get_completion(prompt: str, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )
    return response

def process_text(text, keep_as_list=False, additional_stopwords: List = []):
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
            and word not in additional_stopwords
        ):  # remove punctuation
            # stemmed_word = stemmer.stem(word)
            text_clean.append(word)
    if keep_as_list:
        return text_clean
    return " ".join(text_clean)


def generate_embeddings_2d(
    key_to_vector_embedding: dict, algorithm: Any = TSNE(), word_limit: int = 20
) -> None:
    """Crea una visualizacion de embeddings"""
    assert (
        isinstance(algorithm, PCA)
        or isinstance(algorithm, TSNE)
        or isinstance(algorithm, UMAP)
    ), "La visualizacion solo funciona con instancias PCA, TSNE o UMAP"

    vectors = []
    labels = []
    for key, vector in key_to_vector_embedding.items():
        vectors.append(vector)
        labels.append(key)

    reduced_2d_data = algorithm.fit_transform(np.array(vectors))
    x, y = reduced_2d_data[:, 0], reduced_2d_data[:, 1]

    return pd.DataFrame({"word": labels, "x": x, "y": y}).head(word_limit)


data = pd.read_csv("dashboard/dash_data/processed_data_news.csv", sep="\t")

data["text_tokenized"] = data["Texto"].apply(lambda x: process_text(x, keep_as_list=False))
label_binarizer = LabelBinarizer()

tf_tokenizer = Tokenizer()
fit_text = [" ".join(data["text_tokenized"])]
tf_tokenizer.fit_on_texts(fit_text)

def text_to_index(text):
    """Convierte un texto a una secuencia de indices"""
    return [ tf_tokenizer.word_index[word] for word in text.split(" ")]


def predict_news(text: str, probs_dict: Dict = {}) -> str:
    
    nn_model = mlflow.tensorflow.load_model(model_uri=f"models:/{MLFLOW_FAKE_NEWS_MODEL_NAME}/Production",)

    tokenized = " ".join([
        word for word in process_text(text.lower()).split(" ")
        if word in list(tf_tokenizer.word_index.keys())
       ])

    vector_ = tf.keras.preprocessing.sequence.pad_sequences( 
          np.array(text_to_index(tokenized)).reshape(1,-1),  maxlen=BEST_PARAMS["MAXLEN"]
       )

    return nn_model.predict(vector_)