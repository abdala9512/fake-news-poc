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

def process_text(text,  keep_as_list = False, additional_stopwords: List = []):
    '''
    Input:
        text: a string containing a text
    Output:
        text_clean: a list of words containing the processed text

    '''
    
    stemmer = SnowballStemmer('spanish')
    stopwords_ = stopwords.words('spanish')
    text_tokens = word_tokenize(text)

    text_clean = []
    for word in text_tokens:
        if (word not in stopwords_ and  # remove stopwords
            word not in string.punctuation and 
            word.isalpha() and word not in additional_stopwords):  # remove punctuation
            #stemmed_word = stemmer.stem(word)
            text_clean.append(word)
    if keep_as_list:
        return text_clean
    return ' '.join(text_clean)

def remove_accent(text: str):
    
    unaccented_text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return unaccented_text

def apply_replacement(dataframe: pd.DataFrame, column: str,
                      replace_values: dict):
    dataframe = dataframe.replace({column: replace_values})
    return dataframe

def basic_cleaning(dataframe: pd.DataFrame, text_cols: list = None, 
                   replacements: dict = None, omit: List = []):
    
    
    object_columns = [ col for col in dataframe.select_dtypes("object").columns if col not in omit]
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
            dataframe = apply_replacement(dataframe=dataframe, 
                                          column=col, replace_values=replace)
    
    return dataframe



def generate_N_grams(text: List[str], ngram: int=1, separator: str = " "):
    temp=zip(*[text[i:] for i in range(0,ngram)])
    ans=[separator.join(ngram) for ngram in temp]
    return ans

##########################################################################################################################################################################################
#######################################################################################LEMATIZACION#######################################################################################
##########################################################################################################################################################################################
def lemmatize_text(text_list: str):
    """Spanish lemmatizer with Stanza
    """
    lemmatizer = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma',)
    
    lemmatized_dict = {}
    for word in text_list:
        word_lemmatized = lemmatizer(word)
        lemmatized_dict[word] = word_lemmatized.sentences[0].words[0].lemma
    return lemmatized_dict

def text_to_lemma(text_list: list, lemmatized_vocabulary: dict):
    return ' '.join([lemmatized_vocabulary[word] for word in text_list] )

##########################################################################################################################################################################################
#######################################################################################LDA#######################################################################################
##########################################################################################################################################################################################
def create_topics(data: pd.DataFrame, text_column: str, method: str = "bow", num_topics: int = 5, n_gram = 1, stopwords: str = [],
                  filename: str = "topic_modeling_anexos", print_lda: bool = False):
    """Crea topicos dado un corpora.
    """
    text = data[text_column].apply(lambda x: generate_N_grams(
            process_text( x ,additional_stopwords=stopwords, keep_as_list=True), n_gram
        )
    )
    dictionary_ = gensim.corpora.Dictionary(text)
    
    bow_corpus = [dictionary_.doc2bow(doc) for doc in text]
    
    if method != "bow":
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model = gensim.models.LdaMulticore(
            corpus_tfidf, num_topics=num_topics, id2word=dictionary_, passes=50, workers=4
        )
    else:
        lda_model = gensim.models.LdaMulticore(
            bow_corpus, num_topics=num_topics, id2word=dictionary_, passes=50, workers=4
        )
        
    if print_lda:
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx+1, topic))
        
    LDAvis_prepared = prepare(lda_model, bow_corpus, dictionary_)
    
    if not os.path.isdir("topic_modeling_results"):
        print("Carpeta 'topic_modeling_results' no encontrada. Creando carpeta de resultados de topic_modeling...")
        os.makedirs("topic_modeling_results")
        
    filename = f"topic_modeling_results/{filename}_{num_topics}_{n_gram}_{method}.html"
    pyLDAvis.save_html(LDAvis_prepared, filename )
    print(f"Reporte {filename} generado.")
        
    return lda_model

##########################################################################################################################################################################################
#######################################################################################VISUALIZACION#######################################################################################
##########################################################################################################################################################################################
def create_wordcloud(data: pd.DataFrame, columns: List, **kwargs ):
    """Crea una nube de palabras con las columnas de un dataframe de pandas
    """
    text_ = []
    for col in data[columns].columns:
        for i in data[col].dropna():
            text_.append(i)
    bag_of_words =  "".join(text_)+" "
    
    wordcloud_plot = WordCloud(
        max_font_size=70, max_words=100, 
        width=800, height=500,
        background_color="white", 
        collocations=False).generate(bag_of_words)
    plt.imshow(wordcloud_plot, interpolation='bilinear',)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    

def visualize_embedding(
    key_to_vector_embedding: dict, 
    algorithm: Any, 
    word_limit: int = 20) -> None:
    """Crea una visualizacion de embeddings
    """
    assert (
        isinstance(algorithm, PCA) or 
        isinstance(algorithm, TSNE) or
        isinstance(algorithm, UMAP)
    ), "La visualizacion solo funciona con instancias PCA, TSNE o UMAP"
    
    vectors = []
    labels = []
    for key, vector in key_to_vector_embedding.items():
        vectors.append(vector)
        labels.append(key)
        
    reduced_2d_data = algorithm.fit_transform(np.array(vectors))
    x, y =  reduced_2d_data[:, 0], reduced_2d_data[:, 1]
    
    for i in range(word_limit):
        plt.scatter(x[i],y[i], color="#59C1BD")
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
        )
    plt.xlabel("Dim 1", size=15)
    plt.ylabel("Dim 2", size=15)
    plt.title("Representacion Embeddings",size=30)