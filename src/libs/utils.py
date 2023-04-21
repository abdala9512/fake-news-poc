"""Funciones NLP DNP"""

import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from collections import Counter
import unicodedata
import stanza
from typing import List, Any
import os

import gensim
from gensim import corpora, models
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


import pyLDAvis
from pyLDAvis.gensim_models import prepare


##########################################################################################################################################################################################
#############################################################################PROCESAMIENTO DE TEXTO#######################################################################################
##########################################################################################################################################################################################


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


def remove_accent(text: str):

    unaccented_text = (
        unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    )
    return unaccented_text


def apply_replacement(dataframe: pd.DataFrame, column: str, replace_values: dict):
    dataframe = dataframe.replace({column: replace_values})
    return dataframe


def basic_cleaning(
    dataframe: pd.DataFrame,
    text_cols: list = None,
    replacements: dict = None,
    omit: List = [],
):

    object_columns = [
        col for col in dataframe.select_dtypes("object").columns if col not in omit
    ]
    # Strip text
    for col in object_columns:
        dataframe[col] = dataframe[col].str.strip()
        dataframe[col] = dataframe[col].str.lower()

    # Remove tildes
    if text_cols:
        for col in dataframe[text_cols].columns:
            dataframe[col] = dataframe[col].apply(lambda x: remove_accent(x))

    # Apply replacements
    if replacements:
        for col, replace in replacements.items():
            dataframe = apply_replacement(
                dataframe=dataframe, column=col, replace_values=replace
            )

    return dataframe


def generate_N_grams(text: List[str], ngram: int = 1, separator: str = " "):
    temp = zip(*[text[i:] for i in range(0, ngram)])
    ans = [separator.join(ngram) for ngram in temp]
    return ans


##########################################################################################################################################################################################
#######################################################################################LEMATIZACION#######################################################################################
##########################################################################################################################################################################################
def lemmatize_text(text_list: str):
    """Spanish lemmatizer with Stanza"""
    lemmatizer = stanza.Pipeline(
        lang="es",
        processors="tokenize,mwt,pos,lemma",
    )

    lemmatized_dict = {}
    for word in text_list:
        word_lemmatized = lemmatizer(word)
        lemmatized_dict[word] = word_lemmatized.sentences[0].words[0].lemma
    return lemmatized_dict


def text_to_lemma(text_list: list, lemmatized_vocabulary: dict):
    return " ".join([lemmatized_vocabulary[word] for word in text_list])


##########################################################################################################################################################################################
#######################################################################################VISUALIZACION#######################################################################################
##########################################################################################################################################################################################
def create_wordcloud(data: pd.DataFrame, columns: List, **kwargs):
    """Crea una nube de palabras con las columnas de un dataframe de pandas"""
    text_ = []
    for col in data[columns].columns:
        for i in data[col].dropna():
            text_.append(i)
    bag_of_words = "".join(text_) + " "

    wordcloud_plot = WordCloud(
        max_font_size=70,
        max_words=100,
        width=800,
        height=500,
        background_color="white",
        collocations=False,
    ).generate(bag_of_words)
    plt.imshow(
        wordcloud_plot,
        interpolation="bilinear",
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
